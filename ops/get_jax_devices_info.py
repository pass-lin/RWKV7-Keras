import os
from functools import lru_cache
from typing import Literal

import triton
import jax


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


device = get_available_device() if get_available_device() != "hip" else "cuda"

is_intel_a770 = False
device = jax.devices()
is_tf32_supported = is_nvidia


def get_all_max_shared_memory():
    return [
        triton.runtime.driver.active.utils.get_device_properties(i)["max_shared_mem"]
        for i in range(len(jax.devices()))
    ]


device_shared_mem_list = get_all_max_shared_memory()


@lru_cache(maxsize=None)
def is_triton_shared_mem_enough(
    max_shared_mem: int = 102400, tensor_idx: int = 0
) -> bool:
    max_shared_memory = device_shared_mem_list[tensor_idx]
    return max_shared_memory >= max_shared_mem


device_capacity = is_triton_shared_mem_enough()
