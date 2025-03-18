import os

os.environ["KERAS_BACKEND"] = 'torch'
os.environ['CUDA_VISIBLE_DEVICES'] = "1" 

from keras import ops
from src.convertor import *
from src.layer import *
from standard_rwkv.rwkv7_layer import *


keras.config.set_dtype_policy("bfloat16")
x = torch.randn(1, 8, args.n_embd).cuda()/10
x = ops.cast(x,dtype="bfloat16")
standard_chnnal_mix = RWKV_CMix_x070(args,0).cuda().bfloat16()

my_chnnal_mix = RWKV7_ChannelMix(args.dim_ffn,dtype="bfloat16")
my_chnnal_mix.build(x.shape)
convert_cmix(my_chnnal_mix,standard_chnnal_mix)
standard_time_mix = RWKV_Tmix_x070(args,0).cuda().bfloat16()
standard_time_mix_out = standard_time_mix(x)
my_time_mix = RWKV7_TimeMix(args.dim_att,args.head_size_a)
my_time_mix.build(x.shape)
convert_tmix(my_time_mix,standard_time_mix)
for i in range(1):
    print("第%d次检查是否通过"%i)
    x = x*10
    x = torch.randn(1, 8, args.n_embd).cuda()/10
    x = ops.ones([1, 8, args.n_embd])
    x = ops.cast(x,dtype="bfloat16")
    stanard_cmix_out = standard_chnnal_mix(x)
    my_cmix_out = my_chnnal_mix(x)
    cmix_is_close = ops.isclose(stanard_cmix_out,my_cmix_out,atol=1e-4)
    cmix_is_close = bool(ops.all(cmix_is_close))
    print(f"channal mix check flag :{cmix_is_close}")
    
    mask=np.ones(x.shape[:2])
    my_time_mix_out = my_time_mix(x,mask=mask)
    standard_time_mix_out = standard_time_mix(x)
    time_mix_is_close = bool(ops.all(ops.isclose(my_time_mix_out[0],standard_time_mix_out[0], atol=1e-4)))
    v_first_is_close = bool(ops.all(ops.isclose(my_time_mix_out[1],standard_time_mix_out[1], atol=1e-5)))
    print(f"tmix check flag :{time_mix_is_close}")
    print(f"v_first check flag :{v_first_is_close}")
    
    new_x = ops.concatenate([x,x],1)
    new_mask = ops.concatenate([ops.zeros_like(mask),mask],1)
    my_time_mix_out = my_time_mix(new_x,mask=new_mask)
    time_mix_is_close = bool(ops.all(ops.isclose(my_time_mix_out[0][:,mask.shape[-1]:],standard_time_mix_out[0], atol=1e-4)))
    print(f"tmix check flag add mask :{time_mix_is_close}")
