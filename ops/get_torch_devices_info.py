import functools
import os
from functools import lru_cache
from typing import Literal

import triton
from packaging import version
import torch


@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)[
        "multiprocessor_count"
    ]


@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        import warnings

        warnings.warn(
            ("Triton is not supported on current platform, roll back to CPU."),
            stacklevel=1,
        )
        return "cpu"


@lru_cache(maxsize=None)
def _check_platform() -> Literal["nvidia", "amd", "intel", "musa"]:
    device = get_available_device()
    if device == "cuda":
        return "nvidia"
    elif device == "hip":
        return "amd"
    elif device == "xpu":
        return "intel"
    else:
        return device


# For AMD GPUs, the triton backend is 'hip', while for Nvidia GPUs, the triton backend is 'cuda'.
# However, the torch backend is 'cuda' for both Nvidia and AMD GPUs.
# Therefore, we need to check the triton backend to determine the actual GPU vendor.
device = get_available_device() if get_available_device() != "hip" else "cuda"

device_platform = _check_platform()

is_intel = device_platform == "intel"
is_nvidia = device_platform == "nvidia"
is_amd = device_platform == "amd"

use_cuda_graph = is_nvidia and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"


@lru_cache(maxsize=None)
def check_pytorch_version(version_s: str = "2.4") -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)


is_intel_a770 = is_intel and "Intel(R) Arc(TM) A" in torch.xpu.get_device_name(0)
device = get_available_device() if get_available_device() != "hip" else "cuda"
device_torch_lib = getattr(torch, device)
if check_pytorch_version("2.4"):
    device = "cuda" if device == "cpu" else device
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)

    def custom_device_ctx(index: int):
        return device_torch_lib.device(index)
else:
    assert device == "cuda", (
        "Only cuda device is supported for PyTorch version < 2.4.0."
    )
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd

    def custom_device_ctx(index: int):
        return torch.cuda.device(index)


# Nvidia Ampere or newer, haven't check AMD and intel yet.
is_tf32_supported = is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8


def get_all_max_shared_memory():
    return [
        triton.runtime.driver.active.utils.get_device_properties(i)["max_shared_mem"]
        for i in range(device_torch_lib.device_count())
    ]


device_shared_mem_list = get_all_max_shared_memory()


@lru_cache(maxsize=None)
def is_triton_shared_mem_enough(
    max_shared_mem: int = 102400, tensor_idx: int = 0
) -> bool:
    max_shared_memory = device_shared_mem_list[tensor_idx]
    return max_shared_memory >= max_shared_mem


device_capacity = is_triton_shared_mem_enough()
