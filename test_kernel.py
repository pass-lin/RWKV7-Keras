import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from keras import ops
from ops.native_keras_op import generalized_delta_rule as keras_op
from ops.torch_op import generalized_delta_rule as triton_op
from standard_rwkv.rwkv7_layer_demo import RWKV7_OP as native_op

T = 1024
native_x = np.random.random([1, T, 4 * 64])
x = torch.from_numpy(native_x).cuda().float()


def normalize(
    z,
    p=2,
    dim=-1,
    eps: float = 1e-12,
):
    # F.normalize like api
    denom = ops.norm(z, ord=p, axis=dim, keepdims=True)
    denom = ops.maximum(denom, 1e-12)
    return z / denom


native_output =  native_op(
        x, -ops.softplus(x * 1), x * 2, x * 3, -normalize(x * 4), normalize(x * 4)
    )

x = ops.convert_to_tensor(native_x, dtype="float32")
z = ops.reshape(x, (1, T, 4, 64))
keras_output, state = keras_op(
    z,
    -ops.softplus(z * 1),
    z * 2,
    z * 3,
    -ops.reshape(normalize(x * 4), (1, T, 4, 64)),
    ops.reshape(normalize(x * 4), (1, T, 4, 64)),
)
keras_output = ops.reshape(keras_output, native_output.shape)
keras_is_close = ops.all(ops.isclose(native_output, keras_output, rtol=1e-7))
print(f"keras op check flag :{keras_is_close}")

triton_output, state = triton_op(
    z,
    -ops.softplus(z * 1),
    z * 2,
    z * 3,
    -ops.reshape(normalize(x * 4), (1, T, 4, 64)),
    ops.reshape(normalize(x * 4), (1, T, 4, 64)),
)
triton_output = triton_output.float()
triton_output = ops.reshape(triton_output, native_output.shape)

triton_torch_is_close = ops.all(ops.isclose(native_output, triton_output, rtol=5e-3))
print(f"triton and torch op check flag :{triton_torch_is_close}")
triton_keras_is_close = ops.all(ops.isclose(triton_output, keras_output, rtol=5e-3))
print(f"triton and keras op check flag :{triton_keras_is_close}")
