import keras

if keras.config.backend() == "torch":
    import torch

    if torch.cuda.is_available():
        from ops.torch_op import generalized_delta_rule
    else:
        from ops.native_keras_op import generalized_delta_rule
elif keras.config.backend() == "jax":
    from ops.get_jax_devices_info import is_nvidia
    from jax.lib import xla_bridge

    if is_nvidia and xla_bridge.get_backend().platform == "gpu":
        from jax_op import generalized_delta_rule
    else:
        from ops.native_keras_op import generalized_delta_rule
else:
    from ops.native_keras_op import generalized_delta_rule
