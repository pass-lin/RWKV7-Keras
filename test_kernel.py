from standard_rwkv.rwkv7_layer import RWKV7_OP as native_op
from ops.torch_op import RWKV7_OP as triton_op
import torch
x = torch.randn(1, 8, 256).cuda()
