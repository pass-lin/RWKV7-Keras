import keras

if keras.config.backend() == "torch":
    from ops.torch_op import RWKV7_OP
else:
    from ops.native_keras_op import RWKV7_OP
