import os
os.environ["KERAS_BACKEND"] = 'numpy'
os.environ['CUDA_VISIBLE_DEVICES'] = "1" 
from standard_rwkv.rwkv7_layer import RWKV7_OP as native_op
from ops.native_keras_op import RWKV7_OP as keras_op
import torch
import numpy as np
from keras import ops
native_x = np.random.random([1,8,4*64])
x = torch.from_numpy(native_x).bfloat16()
native_output = native_op(x, x, x, x, x, x).float().cpu().numpy()
x = ops.convert_to_tensor(native_x,dtype="bfloat16")
z = ops.reshape(x,(1,8,4,64))
keras_output = keras_op(z, x, z, z, z, z)
keras_output = ops.convert_to_numpy(ops.cast(keras_output,"float32"))
keras_is_close = ops.all(ops.isclose(native_output, keras_output, atol=1e-5))
print(f"keras op check flag :{keras_is_close}")
