# -*- coding: utf-8 -*-

import functools
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple


import jax
import jax.numpy as jnp
import triton


def tensor_cache(
    fn: Callable[..., jax.Array],
) -> Callable[..., jax.Array]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., jax.Array]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., jax.Array]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


@tensor_cache
def prepare_lens(offsets):
    return offsets[1:] - offsets[:-1]


@tensor_cache
def prepare_chunk_offsets(offsets: jax.Array, chunk_size: int):
    return jnp.cumsum(
        jnp.concat(
            [
                offsets.new_tensor([0]),
                triton.cdiv(prepare_lens(offsets), chunk_size),
            ]
        ),
        axis=-1,
    )
