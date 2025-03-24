import keras

if keras.config.backend() == "torch":
    import torch

    if torch.cuda.is_available():
        from ops.torch_op import generalized_delta_rule
    else:
        from ops.native_keras_op import generalized_delta_rule

else:
    from ops.native_keras_op import generalized_delta_rule
