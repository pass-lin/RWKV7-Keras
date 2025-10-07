# check_layerwise.py
# 每层内部把 TMIX 与 CMIX 所有中间张量拉出来对齐
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 按需改
os.environ["KERNEL_TYPE"] = "native"
import numpy as np
import torch, keras
from keras import ops
from standard_rwkv.rwkv7_layer_demo import RWKV, args, RWKV7_OP  # 标准模型
from src.backbone import RWKV7Backbone
from src.convertor import convert_backbone
from torch.nn import functional as F

# ---------- 基础配置 ----------
args.n_layer = 12
args.vocab_size = 50304
keras.config.set_dtype_policy("bfloat16")
ATOL = 1e-4

# ---------- 构造模型 ----------
std_model = RWKV(args).cuda().bfloat16()
my_model = RWKV7Backbone(
    hidden_size=args.n_embd,
    head_size=args.head_size_a,
    intermediate_dim=args.dim_ffn,
    num_layers=args.n_layer,
    vocabulary_size=args.vocab_size,
)
my_model.eval()
convert_backbone(my_model, std_model)  # 权重对齐

# ---------- 输入 ----------
B, T = 2, 8
x_np = (np.arange(B * T) % 100).astype("int32").reshape(B, T)
x_torch = torch.tensor(x_np, dtype=torch.int32).cuda()
x_keras = ops.convert_to_tensor(x_np, dtype="int32")


# ---------- helper ----------
def log_diff(a, b, name):
    a, b = ops.convert_to_numpy(a), ops.convert_to_numpy(b)
    print(
        f"{name:50s}  max={float(np.max(np.abs(a - b))):.2e}  sum={float(np.sum(np.abs(a - b))):.2e}"
    )


# ---------- 1. embedding ----------
std_emb = std_model.emb(x_torch)
my_emb = my_model.token_embedding(x_keras)
log_diff(my_emb, std_emb, "embedding")

# ---------- 2. 逐层 ----------
v_first = None


##################################################################
# 内部复查函数
##################################################################
def _check_tmix_inside(std_att, my_att, std_x16, my_x16, v_first):
    """bfloat16 级逐张量对比 TimeMix"""
    B, T, C = std_x16.shape
    H, N = std_att.n_head, std_att.head_size

    with torch.no_grad():
        std_xx = std_att.time_shift(std_x16) - std_x16
    my_xx = my_att.time_shift(my_x16) - my_x16
    log_diff(my_xx, std_xx, "tmix xx")

    # xr/xw/xk/xv/xa/xg
    def _shift(x, xx, fx):
        return x + xx * fx

    with torch.no_grad():
        std_xr = _shift(std_x16, std_xx, std_att.x_r)
        std_xw = _shift(std_x16, std_xx, std_att.x_w)
        std_xk = _shift(std_x16, std_xx, std_att.x_k)
        std_xv = _shift(std_x16, std_xx, std_att.x_v)
        std_xa = _shift(std_x16, std_xx, std_att.x_a)
        std_xg = _shift(std_x16, std_xx, std_att.x_g)
    my_xr = _shift(my_x16, my_xx, my_att.x_r)
    my_xw = _shift(my_x16, my_xx, my_att.x_w)
    my_xk = _shift(my_x16, my_xx, my_att.x_k)
    my_xv = _shift(my_x16, my_xx, my_att.x_v)
    my_xa = _shift(my_x16, my_xx, my_att.x_a)
    my_xg = _shift(my_x16, my_xx, my_att.x_g)
    for nm in ["xr", "xw", "xk", "xv", "xa", "xg"]:
        log_diff(eval(f"my_{nm}"), eval(f"std_{nm}"), f"tmix {nm}")

    # r/w/k/v/a/g
    with torch.no_grad():
        std_r = std_att.receptance(std_xr)
        std_w = (
            -F.softplus(-(std_att.w0 + torch.tanh(std_xw @ std_att.w1) @ std_att.w2))
            - 0.5
        )
        std_k = std_att.key(std_xk)
        std_v = std_att.value(std_xv)
        std_a = torch.sigmoid(std_att.a0 + torch.tanh(std_xa @ std_att.a1) @ std_att.a2)
        std_g = torch.sigmoid(std_xg @ std_att.g1) @ std_att.g2
    my_r = my_att.receptance(my_xr)
    my_w = (
        -ops.softplus(-(my_att.w0 + ops.tanh(ops.matmul(my_xw, my_att.w1)) @ my_att.w2))
        - 0.5
    )
    my_k = my_att.key(my_xk)
    my_v = my_att.value(my_xv)
    my_a = ops.sigmoid(my_att.a0 + ops.tanh(ops.matmul(my_xa, my_att.a1)) @ my_att.a2)
    my_g = ops.sigmoid(ops.matmul(my_xg, my_att.g1)) @ my_att.g2
    for nm in ["r", "w", "k", "v", "a", "g"]:
        log_diff(eval(f"my_{nm}"), eval(f"std_{nm}"), f"tmix {nm}")

    # kk normalize
    with torch.no_grad():
        std_kk = std_att.k_k * std_k
        std_kk = F.normalize(std_kk.view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        std_k = std_k * (1 + (std_a - 1) * std_att.k_a)
    my_kk = my_att.k_k * my_k
    my_kk = ops.reshape(my_kk, (B, T, H, N))
    my_kk = F.normalize(my_kk, dim=-1, p=2.0)
    my_kk = ops.reshape(my_kk, (B, T, C))
    my_k = my_k * (1 + (my_a - 1) * my_att.k_a)
    log_diff(my_kk, std_kk, "tmix kk (normalized)")
    log_diff(my_k, std_k, "tmix k (final)")

    # RWKV7_OP
    std_rwk_in = (
        std_r.view(B, T, H * N),
        std_w.view(B, T, H * N),
        std_k.view(B, T, H * N),
        std_v.view(B, T, H * N),
        -std_kk.view(B, T, H * N),
        (std_kk * std_a).view(B, T, H * N),
    )
    my_rwk_in = (
        ops.reshape(my_r, (B, T, H, N)),
        ops.reshape(my_w, (B, T, H, N)),
        ops.reshape(my_k, (B, T, H, N)),
        ops.reshape(my_v, (B, T, H, N)),
        ops.reshape(-my_kk, (B, T, H, N)),
        ops.reshape(my_kk * my_a, (B, T, H, N)),
    )
    for i, (m, s) in enumerate(zip(my_rwk_in, std_rwk_in)):
        log_diff(m.view(B, T, H * N), s.view(B, T, H * N), f"tmix RWKV7_OP input[{i}]")

    with torch.no_grad():
        std_out = RWKV7_OP(*std_rwk_in).view(B, T, C)
    # std_out = std_out.bfloat16()
    my_out, _ = my_att.RWKV7_OP(*my_rwk_in)
    my_out = ops.reshape(my_out, (B, T, C))
    log_diff(my_out, std_out, "tmix RWKV7_OP out (before LN)")

    # LN / gate / output
    with torch.no_grad():
        std_ln = std_att.ln_x(std_out.view(B * T, C)).view(B, T, C)
        std_gated = std_out * std_g
        std_final = std_att.output(std_gated)
    std_ln = std_blk.att.ln_x(std_out.view(B * T, C)).view(B, T, C)
    my_ln = my_blk.att.ln_x(ops.reshape(my_out, (B * T, C)))
    my_ln = ops.reshape(my_ln, (B, T, C))
    my_gated = my_out * my_g
    my_final = my_att.output_layer(my_gated)
    log_diff(my_ln, std_ln, "tmix GroupNorm out")
    log_diff(my_gated, std_gated, "tmix gate*out")
    log_diff(my_final, std_final, "tmix output proj")


def _check_cmix_inside(std_ffn, my_ffn, std_x16, my_x16):
    """bfloat16 级逐张量对比 ChannelMix"""
    # time-shift
    with torch.no_grad():
        std_xx = std_ffn.time_shift(std_x16) - std_x16
    my_xx = my_ffn.time_shift(my_x16) - my_x16
    log_diff(my_xx, std_xx, "cmix xx")

    # kx
    with torch.no_grad():
        std_kx = std_x16 + std_xx * std_ffn.x_k
    my_kx = my_x16 + my_xx * my_ffn.x_k
    log_diff(my_kx, std_kx, "cmix kx")

    # key proj + relu^2
    with torch.no_grad():
        std_key = std_ffn.key(std_kx)
        std_act = torch.relu(std_key) ** 2
    my_key = my_ffn.key(my_kx)
    my_act = ops.relu(my_key) ** 2
    log_diff(my_act, std_act, "cmix act (relu^2)")

    # value proj
    with torch.no_grad():
        std_out = std_ffn.value(std_act)
    my_out = my_ffn.value(my_act)
    log_diff(my_out, std_out, "cmix out (final)")


for lyr_idx in range(args.n_layer):
    print(f"\n========== layer {lyr_idx} ==========")
    std_blk = std_model.blocks[lyr_idx]
    my_blk = my_model.rwkv_layers[lyr_idx]

    # ---- 2.1 输入 ----
    log_diff(my_emb, std_emb, f"layer{lyr_idx} input")

    # ---- 2.2 ln0（仅第0层）----
    if lyr_idx == 0:
        std_emb = std_blk.ln0(std_emb)
        my_emb = my_blk.ln0(my_emb)
        log_diff(my_emb, std_emb, "ln0")

    # ---- 2.3 TimeMix 前 LN ----
    std_x = std_blk.ln1(std_emb)
    my_x = my_blk.ln1(my_emb)
    log_diff(my_x, std_x, "ln1")

    # ---- 2.4 TimeMix 内部复查 ----
    print("---- TimeMix detail ----")
    _check_tmix_inside(
        std_blk.att, my_blk.att, ops.copy(std_x), ops.copy(my_x), v_first
    )
    print("---- TimeMix detail end ----")

    # ---- 2.5 TimeMix 调用 ----
    std_x, std_v_first = std_blk.att(std_x, v_first)
    my_x, my_v_first = my_blk.att(my_x, v_first=v_first, padding_mask=None)
    log_diff(my_x, std_x, "TimeMix out")
    log_diff(my_v_first, std_v_first, "v_first")

    # ---- 2.6 残差 ----
    std_emb = std_emb + std_x
    my_emb = my_emb + my_x
    log_diff(my_emb, std_emb, "residual after att")

    # ---- 2.7 ChannelMix 前 LN ----
    std_x = std_blk.ln2(std_emb)
    my_x = my_blk.ln2(my_emb)
    log_diff(my_x, std_x, "ln2")

    # ---- 2.8 ChannelMix 内部复查 ----
    print("---- ChannelMix detail ----")
    _check_cmix_inside(std_blk.ffn, my_blk.ffn, ops.copy(std_x), ops.copy(my_x))
    print("---- ChannelMix detail end ----")

    # ---- 2.9 ChannelMix 调用 ----
    std_x = std_blk.ffn(std_x)
    my_x = my_blk.ffn(my_x)
    log_diff(my_x, std_x, "ChannelMix out")

    # ---- 2.10 残差 ----
    std_emb = std_emb + std_x
    my_emb = my_emb + my_x
    log_diff(my_emb, std_emb, "residual after ffn")

    v_first = my_v_first

# ---------- 3. 输出 LN ----------
std_out = std_model.ln_out(std_emb)
my_out = my_model.output_layer_norm(my_emb)
log_diff(my_out, std_out, "final ln_out")
