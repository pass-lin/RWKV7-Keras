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
    import jax
    import os

    os.environ["TRITON_LOG_LEVEL"] = "ERROR"  # 只显示错误级别的日志
    os.environ["TRITON_DISABLE_AUTOTUNE"] = "1"  # 禁用自动调优日志
    if is_nvidia and xla_bridge.get_backend().platform == "gpu":
        from ops.jax_op import generalized_delta_rule
    else:
        from ops.native_keras_op import generalized_delta_rule

else:
    from ops.native_keras_op import generalized_delta_rule
