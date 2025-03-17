import os
os.environ["KERAS_BACKEND"] = 'torch'
from keras import ops
from src.layer import *
from standard_rwkv.rwkv7_layer import *
import torch.nn.init as init
keras.config.set_dtype_policy("bfloat16")
x = torch.randn(1, 8, args.n_embd).cuda()/10
x = ops.cast(x,dtype="bfloat16")
standard_chnnal_mix = RWKV_CMix_x070(args,0).cuda().bfloat16()

my_chnnal_mix = RWKV7_ChannelMix(args.dim_ffn,dtype="bfloat16")
my_chnnal_mix(x)

#权重同步
key_weights = standard_chnnal_mix.key.weight.detach().cpu().T
value_weights = standard_chnnal_mix.value.weight.detach().cpu().T
xk_weights = standard_chnnal_mix.x_k.detach().cpu()
my_chnnal_mix.set_weights([xk_weights,key_weights,value_weights,])

stanard_cmix_out = standard_chnnal_mix(x)
my_cmix_out = my_chnnal_mix(x)
cmix_is_close = ops.isclose(stanard_cmix_out,my_cmix_out,atol=1e-4)
cmix_is_close = bool(ops.all(cmix_is_close))
print(f"channal mix check flag :{cmix_is_close}")


standard_time_mix = RWKV_Tmix_x070(args,0).cuda().bfloat16()
standard_time_mix_out = standard_time_mix(x)

my_time_mix = RWKV7_TimeMix(args.dim_att,args.head_size_a)
my_time_mix_out = my_time_mix(x)

#权重同步
weights= [
    standard_time_mix.x_r.detach().cpu(),
    standard_time_mix.x_w.detach().cpu(),
    standard_time_mix.x_k.detach().cpu(),
    standard_time_mix.x_v.detach().cpu(),
    standard_time_mix.x_a.detach().cpu(),
    standard_time_mix.x_g.detach().cpu(),
    
    standard_time_mix.w0.detach().cpu(),
    standard_time_mix.w1.detach().cpu(),
    standard_time_mix.w2.detach().cpu(),
    
    standard_time_mix.a0.detach().cpu(),
    standard_time_mix.a1.detach().cpu(),
    standard_time_mix.a2.detach().cpu(),
    
    standard_time_mix.v0.detach().cpu(),
    standard_time_mix.v1.detach().cpu(),
    standard_time_mix.v2.detach().cpu(),
    
    standard_time_mix.g1.detach().cpu(),
    standard_time_mix.g2.detach().cpu(),
    
    standard_time_mix.k_k.detach().cpu(),
    standard_time_mix.k_a.detach().cpu(),
    standard_time_mix.r_k.detach().cpu(),
    
    standard_time_mix.receptance.weight.detach().cpu().T,
    standard_time_mix.key.weight.detach().cpu().T,
    standard_time_mix.value.weight.detach().cpu().T,
    standard_time_mix.output.weight.detach().cpu().T,
    
    ops.reshape(standard_time_mix.ln_x.weight.detach().cpu(),[-1,args.head_size_a]),
    ops.reshape(standard_time_mix.ln_x.bias.detach().cpu(),[-1,args.head_size_a]),
    
    ]
my_time_mix.set_weights(weights)
x = x*100
my_time_mix_out = my_time_mix(x)
standard_time_mix_out = standard_time_mix(x)
time_mix_is_close = bool(ops.all(ops.isclose(my_time_mix_out[0],standard_time_mix_out[0])))
v_first_is_close = bool(ops.all(ops.isclose(my_time_mix_out[1],standard_time_mix_out[1])))
print(f"tmix check flag :{time_mix_is_close}")#还没对齐，还要测试哪里有问题
print(f"v_first check flag :{v_first_is_close}")