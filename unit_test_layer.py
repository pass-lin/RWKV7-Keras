import os
os.environ["KERAS_BACKEND"] = 'torch'
from keras import ops


from standard_rwkv.rwkv7_layer import *

#x = torch.randn(1, 1, args.n_embd)
standard_time_mix = RWKV_Tmix_x070(args,0)