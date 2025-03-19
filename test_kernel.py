import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch

from keras import ops
from ops.native_keras_op import RWKV7_OP as keras_op
from standard_rwkv.rwkv7_layer_demo import RWKV7_OP as native_op
from ops.torch_op import RWKV7_OP as triton_op
native_x = np.random.random([1, 8, 4 * 64])
x = torch.from_numpy(native_x).cuda().float()
native_output = (
    native_op(x, x + 0.1, x + 0.2, x + 0.3, x + 0.4, x + 0.5)
    .float()
    .cpu()
    .numpy()
)
x = ops.convert_to_tensor(native_x, dtype="float32")
z = ops.reshape(x, (1, 8, 4, 64))
keras_output,state = keras_op(z, z + 0.1, z + 0.2, z + 0.3, z + 0.4, z + 0.5)
keras_output = ops.reshape(keras_output, native_output.shape)
keras_output = ops.convert_to_numpy(ops.cast(keras_output, "float32"))
keras_is_close = ops.all(ops.isclose(native_output, keras_output, rtol=1e-2))
print(f"keras op check flag :{keras_is_close}")

triton_output,state = triton_op(
    z, z + 0.1, z + 0.2, z + 0.3, z + 0.4, z + 0.5
)   
triton_output = triton_output.float().cpu().numpy()
triton_output = np.reshape(triton_output, native_output.shape)
from fla.ops.rwkv7 import chunk_rwkv7
triton_torch_is_close = ops.all(ops.isclose(native_output, triton_output, rtol=1e-2))
print(f"triton and torch op check flag :{triton_torch_is_close}")
triton_keras_is_close = ops.all(ops.isclose(triton_output, keras_output, rtol=1e-2))
print(f"triton and keras op check flag :{triton_keras_is_close}")