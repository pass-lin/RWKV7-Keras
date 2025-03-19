import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras import ops
from src.convertor import *
from src.layer import *
from src.backbone import *
from standard_rwkv.rwkv7_layer_demo import *

args.num_layers = 3
keras.config.set_dtype_policy("bfloat16")

standard_RWKV = RWKV(args)
if os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
    standard_RWKV = standard_RWKV.cuda().bfloat16()
my_backbone = RWKV7Backbone(
    hidden_size=args.n_embd,
    head_size=args.head_size_a,
    intermediate_dim=args.dim_ffn,
    num_layers=args.n_layer,
    vocabulary_size=args.vocab_size,
)
convert_backbone(my_backbone, standard_RWKV)
my_chnnal_mix = my_backbone.rwkv_layers[0].ffn
standard_chnnal_mix = standard_RWKV.blocks[0].ffn


my_time_mix = my_backbone.rwkv_layers[0].att
standard_time_mix = standard_RWKV.blocks[0].att


for i in range(1):
    print("第%d次检查是否通过" % i)
    x = torch.randn(1, 8, args.n_embd).cuda() / 10

    x = ops.cast(x, dtype="bfloat16")
    stanard_cmix_out = standard_chnnal_mix(x)
    my_cmix_out = my_chnnal_mix(x)
    cmix_is_close = ops.isclose(stanard_cmix_out, my_cmix_out, atol=1e-4)
    cmix_is_close = bool(ops.all(cmix_is_close))
    print(f"channal mix check flag :{cmix_is_close}")

    mask = np.ones(x.shape[:2])
    my_time_mix_out = my_time_mix(x, padding_mask=mask)
    standard_time_mix_out = standard_time_mix(x)
    time_mix_is_close = bool(
        ops.all(
            ops.isclose(my_time_mix_out[0], standard_time_mix_out[0], atol=1e-4)
        )
    )
    v_first_is_close = bool(
        ops.all(
            ops.isclose(my_time_mix_out[1], standard_time_mix_out[1], atol=1e-4)
        )
    )

    print(f"tmix check flag :{time_mix_is_close}")
    print(f"v_first check flag :{v_first_is_close}")

    new_x = ops.concatenate([x, x], 1)
    new_mask = ops.concatenate([ops.zeros_like(mask), mask], 1)
    my_time_mix_out = my_time_mix(new_x, padding_mask=new_mask)
    time_mix_is_close = bool(
        ops.all(
            ops.isclose(
                my_time_mix_out[0][:, mask.shape[-1] :],
                standard_time_mix_out[0],
                atol=1e-4,
            )
        )
    )
    print(f"tmix check flag add mask :{time_mix_is_close}")
    v_firts = ops.cast(torch.randn(1, 8, args.n_embd).cuda(), dtype="bfloat16")
    x = ops.cast(torch.randn(1, 8, args.n_embd).cuda(), dtype="bfloat16")

    print("第一层block test")
    my_block = my_backbone.rwkv_layers[0]
    standard_block = standard_RWKV.blocks[0]
    standard_block_out = standard_block(x, v_firts)
    my_block_out = my_block.call(x)
    block_is_close = bool(
        ops.all(ops.isclose(my_block_out[0], standard_block_out[0], atol=1e-1))
    )
    block_v_first_is_close = bool(
        ops.all(ops.isclose(my_block_out[1], standard_block_out[1], atol=1e-1))
    )

    print(f"block check flag :{block_is_close}")
    print(f"block v_first check flag :{block_v_first_is_close}")

    print("第二层block test")
    my_block = my_backbone.rwkv_layers[1]
    standard_block = standard_RWKV.blocks[1]
    standard_block_out = standard_block(x, v_firts)
    my_block_out = my_block(x, v_firts)
    block_is_close = bool(
        ops.all(ops.isclose(my_block_out[0], standard_block_out[0], atol=1e-1))
    )
    block_v_first_is_close = bool(
        ops.all(ops.isclose(my_block_out[1], standard_block_out[1], atol=1e-2))
    )

    print(f"block check flag :{block_is_close}")
    print(f"block v_first check flag :{block_v_first_is_close}")

    x = ops.arange(16) + i * 10 + 4
    x = ops.reshape(x, [2, 8])

    my_backbone_output = my_backbone(x)
    standard_backbone_output = standard_RWKV(x, True)
    model_is_close = bool(
        ops.all(
            ops.isclose(my_backbone_output, standard_backbone_output, atol=1e-1)
        )
    )
    print(f"model check flag :{model_is_close}")
    if not model_is_close:
        pass
        # print(my_backbone_output-standard_backbone_output)
