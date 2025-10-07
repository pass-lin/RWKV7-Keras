import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["KERNEL_TYPE"] = "native"

import numpy as np
import torch
from keras import ops
from src.backbone import *
from src.convertor import *
from src.layer import *
from standard_rwkv.rwkv7_layer_demo import *
from modelscope import snapshot_download

# ------------------------------------------------------------------
# 统一容差
# ------------------------------------------------------------------
ATOL = 1e-2  # 全局绝对容差，可按需改

# ------------------------------------------------------------------
# 基础配置
# ------------------------------------------------------------------
args.n_layer = 12
dtype_policy = "float32"
keras.config.set_dtype_policy(dtype_policy)

standard_RWKV = RWKV(args)
if os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
    standard_RWKV = standard_RWKV.cuda()
if dtype_policy == "bfloat16":
    standard_RWKV = standard_RWKV.bfloat16()
else:
    standard_RWKV = standard_RWKV.float()
souce_model_name = "RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"

download_path = snapshot_download(
    repo_id="Blink_DL/rwkv-7-world",
    allow_patterns=souce_model_name,
)
weights_path = os.path.join(download_path, souce_model_name)
weights = torch.load(weights_path, map_location="cpu")
standard_RWKV.load_state_dict(weights, strict=False)

my_backbone = RWKV7Backbone(
    hidden_size=args.n_embd,
    head_size=args.head_size_a,
    intermediate_dim=args.dim_ffn,
    num_layers=args.n_layer,
    vocabulary_size=args.vocab_size,
)
my_backbone.eval()
convert_backbone(my_backbone, standard_RWKV)

my_chnnal_mix = my_backbone.rwkv_layers[0].ffn
standard_chnnal_mix = standard_RWKV.blocks[0].ffn

my_time_mix = my_backbone.rwkv_layers[0].att
standard_time_mix = standard_RWKV.blocks[0].att


# ------------------------------------------------------------------
# 开始测试
# ------------------------------------------------------------------
def convert_to_numpy(x):
    x = ops.cast(x, dtype="float32")
    return ops.convert_to_numpy(x)


for i in range(1):
    print("========== 第 %d 次检查 ==========" % i)

    # ---------- channel mix ----------
    x = torch.randn(1, 8, args.n_embd).cuda() / 10
    x = ops.cast(x, dtype=dtype_policy)

    stanard_cmix_out = standard_chnnal_mix(x)
    my_cmix_out = my_chnnal_mix(x)
    np.testing.assert_allclose(
        convert_to_numpy(my_cmix_out), convert_to_numpy(stanard_cmix_out), atol=ATOL
    )
    print("channel mix check passed.")

    # ---------- time mix (无 mask) ----------
    mask = np.ones(x.shape[:2])
    my_time_mix_out = my_time_mix(x, padding_mask=mask)
    standard_time_mix_out = standard_time_mix(x)

    np.testing.assert_allclose(
        convert_to_numpy(my_time_mix_out[0]),
        convert_to_numpy(standard_time_mix_out[0]),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        convert_to_numpy(my_time_mix_out[1]),
        convert_to_numpy(standard_time_mix_out[1]),
        atol=ATOL,
    )
    print("time mix (no mask) check passed.")

    # ---------- time mix (有 mask) ----------
    new_x = ops.concatenate([x, x], 1)
    new_mask = ops.concatenate([ops.zeros_like(mask), mask], 1)
    my_time_mix_out = my_time_mix(new_x, padding_mask=new_mask)

    np.testing.assert_allclose(
        convert_to_numpy(my_time_mix_out[0][:, mask.shape[-1] :]),
        convert_to_numpy(standard_time_mix_out[0]),
        atol=ATOL,
    )
    print("time mix (with mask) check passed.")

    # ---------- block level ----------
    v_first = ops.cast(torch.randn(1, 8, args.n_embd).cuda(), dtype=dtype_policy)
    x = ops.cast(torch.randn(1, 8, args.n_embd).cuda(), dtype=dtype_policy)

    # layer 0
    print("第一层 block test")
    my_block_out = my_backbone.rwkv_layers[0](x)
    standard_block_out = standard_RWKV.blocks[0](x, v_first)

    np.testing.assert_allclose(
        convert_to_numpy(my_block_out[0]),
        convert_to_numpy(standard_block_out[0]),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        convert_to_numpy(my_block_out[1]),
        convert_to_numpy(standard_block_out[1]),
        atol=ATOL,
    )
    print("block 0 check passed.")

    """    # layer 1
    print("第二层 block test")
    my_block_out     = my_backbone.rwkv_layers[1](x, v_first)
    standard_block_out = standard_RWKV.blocks[1](x, v_first)

    np.testing.assert_allclose(
        convert_to_numpy(my_block_out[0]),
        convert_to_numpy(standard_block_out[0]),
        atol=ATOL
    )
    np.testing.assert_allclose(
        convert_to_numpy(my_block_out[1]),
        convert_to_numpy(standard_block_out[1]),
        atol=ATOL
    )
    print("block 1 check passed.")"""

    # ---------- full model ----------
    x = ops.arange(16) + i * 10 + 4
    x = ops.reshape(x, [2, 8])

    my_backbone_output = my_backbone(x)
    standard_backbone_output = standard_RWKV(x)

    np.testing.assert_allclose(
        convert_to_numpy(my_backbone_output),
        convert_to_numpy(standard_backbone_output),
        atol=ATOL,
    )
    print("full model check passed.")
    print(
        "完全相等:",
        np.all(
            convert_to_numpy(my_backbone_output)
            == convert_to_numpy(standard_backbone_output)
        ),
    )
