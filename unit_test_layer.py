import os
os.environ["KERAS_BACKEND"] = 'torch'
from keras import ops
from src.layer import *
from standard_rwkv.rwkv7_layer import *
import torch.nn.init as init

x = torch.randn(1, 8, args.n_embd).cuda()/10
x = ops.cast(x,dtype="bfloat16")
standard_chnnal_mix = RWKV_CMix_x070(args,0).cuda().bfloat16()

my_chnnal_mix = RWKV7_ChannelMix(args.dim_ffn,dtype="bfloat16")
my_chnnal_mix(x)


key_weights = standard_chnnal_mix.key.weight.detach().cpu().T
value_weights = standard_chnnal_mix.value.weight.detach().cpu().T
xk_weights = standard_chnnal_mix.x_k.detach().cpu()
my_chnnal_mix.set_weights([xk_weights,key_weights,value_weights,])

stanard_cmix_out = standard_chnnal_mix(x)
my_cmix_out = my_chnnal_mix(x)
time_mix_is_close = ops.isclose(stanard_cmix_out,my_cmix_out,atol=1e-4)
time_mix_is_close = bool(ops.all(time_mix_is_close))

standard_time_mix = RWKV_Tmix_x070(args,0).cuda().bfloat16()
standard_time_mix_out = standard_time_mix(x)