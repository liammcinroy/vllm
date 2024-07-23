"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import os
from functools import lru_cache, wraps
from typing import Tuple

import pynvml

from .interface import Platform, PlatformEnum


def with_nvml_context(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pynvml.nvmlShutdown()

    return wrapper


@lru_cache(maxsize=8)
@with_nvml_context
def get_physical_device_capability(
    device_id: int | str = 0
) -> Tuple[int, int]:
    if isinstance(device_id, int):
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    else:
        if "MIG" in device_id:
            # we cannot get a handle for a MIG device, but they all have
            # compute capability 9.0 (as of 2024/07/23).
            # See Table 1 of
            # https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html
            return (9, 0)
        handle = pynvml.nvmlDeviceGetHandleByUUID(device_id)
    return pynvml.nvmlDeviceGetCudaComputeCapability(handle)


def device_id_to_physical_device_id(device_id: int) -> int | str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        physical_device_id = device_ids[device_id]
        try:
            return int(physical_device_id)
        except ValueError:
            # a UUID has been supplied
            return physical_device_id
    else:
        return device_id


class CudaPlatform(Platform):
    _enum = PlatformEnum.CUDA

    @staticmethod
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        physical_device_id = device_id_to_physical_device_id(device_id)
        return get_physical_device_capability(physical_device_id)
