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

import pytest

import arrayfire as af
import numpy as np
import dpnp
import cupy
import arrayfire

ROUNDS = 30
ITERATIONS = 1

NSIZE = 2**8 # Array column size

DTYPE = "float32"
#PKGS = [dpnp, np, cupy, af]
PKGS = [np, cupy, arrayfire]
#IDS = [pkg.__name__ for pkg in PKGS]
IDSAF = ["arrayfire", "numpy", "cupy"]
#IDS = ["numpy", "cupy"]
#IDS = ["dpnp", "numpy", "cupy"]

def generate_arrays(pkg, count):
    arr_list = []
    pkg = pkg.__name__
    
    if "cupy" == pkg:
        for i in range(count):
            #arr_list.append(cupy.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))
            arr_list.append(cupy.random.normal(size=NSIZE * NSIZE).reshape((NSIZE, NSIZE)).astype(DTYPE))
        cupy.cuda.runtime.deviceSynchronize()
    elif "arrayfire" == pkg:
        af.device_gc()
        for i in range(count):  
            arr_list.append(af.arange(1, NSIZE * NSIZE + 1, dtype=DTYPE).reshape((NSIZE, NSIZE)))
    elif "dpnp" == pkg:
        for i in range(count):
            arr_list.append(dpnp.arange(1, NSIZE * NSIZE + 1, dtype=DTYPE).reshape((NSIZE, NSIZE)))
    elif "numpy" == pkg:
        for i in range(count):
            #arr_list.append(np.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))
            arr_list.append(np.random.normal(size=NSIZE * NSIZE).reshape((NSIZE, NSIZE)).astype(DTYPE))
    elif "arrayfire" == pkg:
        for i in range(count):
            #arr_list.append(np.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))
            # TODO array-api
            arr_list.append(arrayfire.randu((NSIZE, NSIZE), dtype=arrayfire.f32))

    return arr_list

@pytest.mark.parametrize(
    "pkg", [arrayfire,np,cupy], ids=IDSAF
    #"pkg", [np,cupy], ids=IDS
    #"pkg", [dpnp,np,cupy], ids=IDS
)
class TestFFT:
    def test_fft(self, benchmark, pkg):
        #setup = lambda: (generate_arrays(pkg.fft.fft if pkg == arrayfire else pkg.fft, 1), {})
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.fft if pkg == arrayfire else pkg.fft.fft,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )
