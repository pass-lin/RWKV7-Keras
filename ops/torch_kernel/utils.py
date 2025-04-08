# -*- coding: utf-8 -*-

import contextlib
import functools
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import triton

from ops.get_torch_devices_info import autocast_custom_bwd
from ops.get_torch_devices_info import autocast_custom_fwd
from ops.get_torch_devices_info import custom_device_ctx


def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
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
def prepare_lens(offsets: torch.LongTensor) -> torch.LongTensor:
    return offsets[1:] - offsets[:-1]


@tensor_cache
def prepare_position_ids(offsets: torch.LongTensor) -> torch.LongTensor:
    return torch.cat([torch.arange(n) for n in prepare_lens(offsets).tolist()]).to(
        offsets.device
    )


@tensor_cache
def prepare_sequence_ids(position_ids: torch.LongTensor) -> torch.LongTensor:
    return position_ids.eq(0).cumsum(0) - 1


@tensor_cache
def prepare_token_indices(offsets: torch.LongTensor) -> torch.LongTensor:
    position_ids = prepare_position_ids(offsets)
    return torch.stack([prepare_sequence_ids(position_ids), position_ids], 1).to(
        offsets
    )


@tensor_cache
def prepare_chunk_offsets(offsets: torch.Tensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat(
        [
            offsets.new_tensor([0]),
            triton.cdiv(prepare_lens(offsets), chunk_size),
        ]
    ).cumsum(-1)


@tensor_cache
def prepare_chunk_indices(
    offsets: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(offsets), chunk_size).tolist()
        ]
    )
    return torch.stack([prepare_sequence_ids(indices), indices], 1).to(offsets)


def require_version(version, hint):
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            from transformers.utils.versions import require_version

            require_version(version, hint)
            return fn(
                ctx,
                *(
                    i if not isinstance(i, torch.Tensor) else i.contiguous()
                    for i in args
                ),
                **{
                    k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
                    for k, v in kwargs.items()
                },
            )

        return wrapper

    return decorator


def checkpoint(fn):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)

    return wrapper


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


contiguous = input_guard


def autocast_contiguous_custom_device_fwd(fn: callable) -> callable:
    """
    A decorator that combines the functionality of contiguous and autocast.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_fn = input_guard(fn)
        autocast_contiguous_fn = autocast_custom_fwd(contiguous_fn)
        return autocast_contiguous_fn(*args, **kwargs)

    return wrapper


def autocast_contiguous_custom_device_bwd(fn: callable) -> callable:
    """
    A decorator that combines the functionality of contiguous and autocast.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_fn = input_guard(fn)
        autocast_contiguous_fn = autocast_custom_bwd(contiguous_fn)
        return autocast_contiguous_fn(*args, **kwargs)

    return wrapper
