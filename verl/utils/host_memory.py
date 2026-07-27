# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reclaim host memory left behind by bulk tensor loads."""

import ctypes
import gc
import logging
import os

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_GB = 1 << 30


class _MallInfo2(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


def _libc():
    try:
        return ctypes.CDLL("libc.so.6")
    except OSError:
        return None


def _retained_bytes() -> int:
    """Bytes glibc has freed but kept instead of returning to the OS. 0 when unavailable."""
    lib = _libc()
    if lib is None or not hasattr(lib, "mallinfo2"):
        return 0
    try:
        lib.mallinfo2.restype = _MallInfo2
        return lib.mallinfo2().fordblks
    except Exception:
        return 0


def release_freed_host_memory(tag: str = "", rank: int | None = None) -> None:
    """Collect cyclic garbage and hand glibc's retained arenas back to the OS.

    Bulk loads leave hundreds of GiB in tensors held by reference cycles (Megatron's sharded
    state-dict structures), which reference counting cannot free. CPython's cycle collector can,
    but it triggers on object *counts*, not bytes -- a handful of objects holding that much never
    reaches the threshold, and full collections are additionally suppressed once a process has a
    large long-lived heap. So the collection has to be forced explicitly.

    This matters because Ray's OOM monitor counts anon + shmem and ignores page cache, so
    uncollected tensors read as live memory and get the worker killed. At scale the optimizer
    restore alone can leave hundreds of GiB per node behind, all of which is returned here.

    malloc_trim is a belt-and-braces second step: allocations this large are mmap-backed and
    already return on free, so it is usually inert, but it costs microseconds and helps whenever
    the freed memory did come off the brk heap.
    """
    before = _retained_bytes()
    gc.collect()
    lib = _libc()
    trimmed = False
    if lib is not None and hasattr(lib, "malloc_trim"):
        try:
            lib.malloc_trim.argtypes = [ctypes.c_size_t]
            lib.malloc_trim.restype = ctypes.c_int
            trimmed = bool(lib.malloc_trim(0))
        except Exception as e:
            logger.warning(f"[hostmem] malloc_trim failed: {type(e).__name__}: {e}")
    after = _retained_bytes()
    logger.info(
        "[hostmem][rank %s] release%s gc+malloc_trim(returned=%s) glibc_retained %.1f -> %.1f GiB",
        rank if rank is not None else os.environ.get("RANK", -1),
        f" {tag}" if tag else "",
        trimmed,
        before / _GB,
        after / _GB,
    )
