########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)
import torch.nn.init as init

"""
This will load RWKV-7 "Goose" x070 and inference in GPT-mode (slower than RNN-mode for autoregressive generation)
"""

args = types.SimpleNamespace()

# model download: https://huggingface.co/BlinkDL/rwkv-7-pile

MODEL_PATH = "/mnt/e/RWKV-x070-Pile-168M-20241120-ctx4096.pth"
# MODEL_PATH = "/mnt/program/RWKV-x070-Pile-421M-20241127-ctx4096.pth"

if "168M" in MODEL_PATH:
    args.n_layer = 12
    args.n_embd = 768
    D_DECAY_LORA = 64
    D_AAA_LORA = 64
    D_MV_LORA = 32
    D_GATE_LORA = 128
elif "421M" in MODEL_PATH:
    args.n_layer = 24
    args.n_embd = 1024
    D_DECAY_LORA = 64
    D_AAA_LORA = 64
    D_MV_LORA = 64
    D_GATE_LORA = 128

args.vocab_size = 50304  # "pile" model: 50277 padded to 50304

DTYPE = torch.bfloat16
# DTYPE = torch.half # better

args.head_size_a = 64  # don't change
HEAD_SIZE = args.head_size_a

USE_KERNEL = False  # False => UNOPTIMIZED, VERY SLOW

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script

########################################################################################################
# CUDA Kernel
########################################################################################################


def RWKV7_OP(r, w, k, v, a, b):
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    r = r.view(B, T, H, N).float()
    
    k = k.view(B, T, H, N).float()
    
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk

        out[:, t, :] = (state @ rr).view(B, H, N)
    return out.view(B, T, C)


########################################################################################################
# RWKV TimeMix
########################################################################################################


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        H = self.n_head
        N = self.head_size
        C = args.n_embd

        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))

        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))

        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))

        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)  # !!! notice eps value !!!

        # Initialize all parameters
        init.normal_(self.x_r, mean=0, std=0.02)
        init.normal_(self.x_w, mean=0, std=0.02)
        init.normal_(self.x_k, mean=0, std=0.02)
        init.normal_(self.x_v, mean=0, std=0.02)
        init.normal_(self.x_a, mean=0, std=0.02)
        init.normal_(self.x_g, mean=0, std=0.02)

        init.normal_(self.w0, mean=0, std=0.02)
        init.normal_(self.w1, mean=0, std=0.02)
        init.normal_(self.w2, mean=0, std=0.02)

        init.normal_(self.a0, mean=0, std=0.02)
        init.normal_(self.a1, mean=0, std=0.02)
        init.normal_(self.a2, mean=0, std=0.02)

        init.normal_(self.v0, mean=0, std=0.02)
        init.normal_(self.v1, mean=0, std=0.02)
        init.normal_(self.v2, mean=0, std=0.02)

        init.normal_(self.g1, mean=0, std=0.02)
        init.normal_(self.g2, mean=0, std=0.02)

        init.normal_(self.k_k, mean=0, std=0.02)
        init.normal_(self.k_a, mean=0, std=0.02)
        init.normal_(self.r_k, mean=0, std=0.02)

        # Initialize Linear layers
        init.normal_(self.receptance.weight, mean=0, std=0.02)
        init.normal_(self.key.weight, mean=0, std=0.02)
        init.normal_(self.value.weight, mean=0, std=0.02)
        init.normal_(self.output.weight, mean=0, std=0.02)

        init.normal_(self.ln_x.weight, mean=0, std=0.02)
        init.normal_(self.ln_x.bias, mean=0, std=0.02)  # !!! notice eps value !!!

    def forward(self, x, v_first=None):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = (
            -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        )  # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2
            )  # add value residual
        a = torch.sigmoid(
            self.a0 + (xa @ self.a1) @ self.a2
        )  # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k

        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)

        k = k * (1 + (a - 1) * self.k_a)

        x = RWKV7_OP(r, w, k, v, -kk, kk * a).to(DTYPE)

        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


########################################################################################################
# RWKV ChannelMix
########################################################################################################


class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

        init.normal_(self.key.weight, mean=0, std=0.02)
        init.normal_(self.value.weight, mean=0, std=0.02)
        init.normal_(self.x_k, mean=0, std=0.02)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x

        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


########################################################################################################
# RWKV Block
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        # only used in block 0, should be fused with emb
        self.ln0 = nn.LayerNorm(args.n_embd)
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        xx, v_first = self.att(self.ln1(x), v_first)

        x = x + xx
        x = x + self.ffn(self.ln2(x))

        return x, v_first


########################################################################################################
# RWKV Model
########################################################################################################


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx, return_hidden_state=False):
        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)

        x = self.ln_out(x)
        if return_hidden_state:
            return x
        x = self.head(x)

        return x


args.dim_att = args.n_embd
args.dim_ffn = args.n_embd * 4
