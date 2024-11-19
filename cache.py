# Copyright (c) 2023-2024, Christian Gilli <christian.gilli11@gmail.com>
# All rights reserved.
#
# This code is licensed under the terms of the MIT license
# (https://opensource.org/license/mit).

import math
from dataclasses import dataclass
from typing import Dict
import psutil
import sys
import ctypes


# Please not that I copied these values by hand so they may or may not work on other Linux machines
_SC_LEVEL1_DCACHE_SIZE = 188
_SC_LEVEL1_DCACHE_ASSOC = 189
_SC_LEVEL1_DCACHE_LINESIZE = 190
_SC_LEVEL2_CACHE_SIZE = 191
_SC_LEVEL2_CACHE_ASSOC = 192
_SC_LEVEL2_CACHE_LINESIZE = 193
_SC_LEVEL3_CACHE_SIZE = 194
_SC_LEVEL3_CACHE_ASSOC = 195
_SC_LEVEL3_CACHE_LINESIZE = 196


@dataclass
class _CacheParams:
    l1_line_size: int = 64
    l2_line_size: int = 64
    l3_line_size: int = 64
    l1_assoc: int = 8
    l2_assoc: int = 4
    l3_assoc: int = 12
    l1_size: int = 32 * 1024
    l2_size: int = 256 * 1024
    # Since the L3 is shared between all the physical cores consider only a fraction of it (in my case 1/6th)
    l3_size: int = int(1.5 * 1024 * 1024)

    def __init__(self):
        if 'linux' in sys.platform:
            # Try to use 'sysconf' from libc, if it's not available for some reason use the defaults
            try:
                libc = ctypes.CDLL('libc.so.6')
                self.l1_line_size = libc.sysconf(_SC_LEVEL1_DCACHE_LINESIZE)
                self.l2_line_size = libc.sysconf(_SC_LEVEL2_CACHE_LINESIZE)
                self.l3_line_size = libc.sysconf(_SC_LEVEL3_CACHE_LINESIZE)

                self.l1_assoc = libc.sysconf(_SC_LEVEL1_DCACHE_ASSOC)
                self.l2_assoc = libc.sysconf(_SC_LEVEL2_CACHE_ASSOC)
                self.l3_assoc = libc.sysconf(_SC_LEVEL3_CACHE_ASSOC)

                self.l1_size = libc.sysconf(_SC_LEVEL1_DCACHE_SIZE)
                self.l2_size = libc.sysconf(_SC_LEVEL2_CACHE_SIZE)
                self.l3_size = libc.sysconf(_SC_LEVEL3_CACHE_SIZE) // psutil.cpu_count(logical=False)
            except:
                raise RuntimeWarning('Unable to load libc, using default cache parameters')
        else:
            raise RuntimeWarning('Not on Linux, using default cache parameters. Consider setting them manually for best performance')



def compute_auto(dtype_size: int, mr: int, nr: int, unroll: int = 1) -> Dict[str, int]:
    """
    This function computes the default cache parameters given the underlying hardware.
    For more context refer to: https://dl.acm.org/doi/10.1145/2925987
    """
    cache = _CacheParams()
    l1_n_sets = cache.l1_size / (cache.l1_line_size * cache.l1_assoc)
    l2_n_sets = cache.l2_size / (cache.l2_line_size * cache.l2_assoc)
    l3_n_sets = cache.l3_size / (cache.l3_line_size * cache.l3_assoc)
    
    # Use ceil to be more greedy
    C_ar = math.ceil((cache.l1_assoc - 1) / (1 + nr / mr))
    C_br = math.floor(C_ar * nr / mr)
    assert C_ar + C_br < cache.l1_assoc

    kc = (C_ar * l1_n_sets * cache.l1_line_size) / (mr * dtype_size)
    # Make kc a mutiple of unroll
    kc = math.floor(kc / unroll) * unroll
    # Assert that we have enough space in the L1 for a micropanel of A, a micropanel of B and a microtile of C
    assert (mr * kc + nr * kc + mr * nr) * dtype_size < cache.l1_size

    # MC x KC tile of A must fit in the L2 cache, it's also good if it's a multiple of mr
    mc = math.ceil(((cache.l2_assoc - 1) * l2_n_sets * cache.l2_line_size) / (kc * dtype_size))
    mc = math.ceil(mc / mr) * mr
    assert mc * kc * dtype_size < cache.l2_size

    # MC x NC panel must fit the L3 also, make it a multiple of nr
    nc = math.ceil(((cache.l3_assoc - 1) * l3_n_sets * cache.l3_line_size) / (mc * dtype_size))
    nc = math.ceil(nc / nr) * nr
    assert mc * nc * dtype_size < cache.l3_size

    return {
        'KC': kc,
        'MC': mc,
        'NC': nc,
    }
