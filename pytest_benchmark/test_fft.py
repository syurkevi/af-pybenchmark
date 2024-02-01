# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import numpy as np
import pytest

import dpnp
import cupy

ROUNDS = 30
ITERATIONS = 1

NSIZE = 2**8 # Array column size

DTYPE = "float32"
IDS = ["dpnp", "numpy", "cupy"]

def generate_arrays(function, count):
    arr_list = []

    pkg = None
    try:
        pkg = function.__module__
    except AttributeError:
        pkg = function.__class__.__module__
    
    if "cupy" in pkg:
        for i in range(count):
            arr_list.append(cupy.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))
        cupy.cuda.runtime.deviceSynchronize()
    # elif "arrayfire" in function.__name__:
    #     for i in range(count):  
    #         arr_list.append(arrayfire.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))
    elif "dpnp" in pkg:
        for i in range(count):
            arr_list.append(dpnp.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))
    elif "numpy" in pkg:
        for i in range(count):
            arr_list.append(np.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))

    return arr_list

@pytest.mark.parametrize(
    "pkg", [dpnp,np,cupy], ids=IDS
)
class TestFFT:
    def test_fft(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg.fft.fft, 1), {})

        result = benchmark.pedantic(
            target=pkg.fft.fft,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )