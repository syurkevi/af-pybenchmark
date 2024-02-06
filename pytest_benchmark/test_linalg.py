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

ROUNDS = 30
ITERATIONS = 1

NSIZE = 2**8 # Array column size
NTSIZE = 2**4 # Tensor column size

DTYPE = "float32"
PKGS = [dpnp, np, cupy, af]
IDS = [pkg.__name__ for pkg in PKGS]

def generate_arrays(pkg, count):
    arr_list = []
    pkg = pkg.__name__
    if "cupy" == pkg:
        for i in range(count):
            arr_list.append(cupy.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))
        cupy.cuda.runtime.deviceSynchronize()
    elif "arrayfire" == pkg:
        af.device_gc()
        for i in range(count):  
            arr_list.append(af.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))
    elif "dpnp" == pkg:
        for i in range(count):
            arr_list.append(dpnp.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))
    elif "numpy" == pkg:
        for i in range(count):
            arr_list.append(np.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))

    return arr_list

def generate_tensor(pkg, count):
    arr_list = []
    pkg = pkg.__name__
    if "cupy" == pkg:
        for i in range(count):
            arr_list.append(cupy.arange(0, NTSIZE ** 3, dtype=DTYPE).reshape((NTSIZE, NTSIZE, NTSIZE)))
        cupy.cuda.runtime.deviceSynchronize()
    # elif "arrayfire" == pkg:
    #     for i in range(count):
    #         arr_list.append(arrayfire.arange(0, NTSIZE ** 3, dtype=DTYPE).reshape((NTSIZE, NTSIZE, NTSIZE)))
    elif "dpnp" == pkg:
        for i in range(count):
            arr_list.append(dpnp.arange(0, NTSIZE ** 3, dtype=DTYPE).reshape((NTSIZE, NTSIZE, NTSIZE)))
    elif "numpy" == pkg:
        for i in range(count):
            arr_list.append(np.arange(0, NTSIZE ** 3, dtype=DTYPE).reshape((NTSIZE, NTSIZE, NTSIZE)))
    
    return arr_list

@pytest.mark.parametrize(
    "pkg", PKGS, ids=IDS
)
class Eindot:
    def test_dot_a_b(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 2), {})

        result = benchmark.pedantic(
            target=pkg.dot,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )

    def test_einsum_ij_jk_a_b(self, benchmark, pkg):
        setup = lambda: (["ij,jk", *generate_arrays(pkg, 2)], {})

        result = benchmark.pedantic(
            target=pkg.einsum,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )

    def test_tensordot_a_b(self, benchmark, pkg):
        setup = lambda: (generate_tensor(pkg, 2), {"axes": ([1, 0], [0, 1])})

        result = benchmark.pedantic(
            target=pkg.tensordot,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )

    def test_matmul_a_b(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 2), {})

        result = benchmark.pedantic(
            target=pkg.matmul,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )

    def test_matmul_a_bt(self, benchmark, pkg):
        a, b = generate_arrays(pkg, 2)
        setup = lambda: ([a, b.T], {})

        result = benchmark.pedantic(
            target=pkg.matmul,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )

        
@pytest.mark.parametrize(
    "pkg", PKGS, ids=IDS
)
class TestLinalg:
    # def test_lstsq(self, benchmark, pkg):
    #     a, b = generate_arrays(pkg)
    #     setup = lambda: (generate_arrays(pkg), {"rcond":-1})

    #     result = benchmark.pedantic(
    #         target=pkg.lstsq,
    #         setup=setup,
    #         rounds=ROUNDS,
    #         iterations=ITERATIONS,
    #     )
    def test_cholesky(self, benchmark, pkg):
        arr = generate_arrays(pkg, 1)[0]
        setup = lambda: ([arr + arr.T], {})

        result = benchmark.pedantic(
            target=pkg.linalg.cholesky,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_svd(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.linalg.svd,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )
    
    def test_inv(self, benchmark, pkg):
        arr = generate_arrays(pkg, 1)[0]
        setup = lambda: ([arr @ arr.T], {})

        result = benchmark.pedantic(
            target=pkg.linalg.inv,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    # def test_pinv(self, benchmark, pkg):
    #     setup = lambda: (generate_arrays(pkg, 1), {})

    #     result = benchmark.pedantic(
    #         target=pkg.linalg.pinv,
    #         setup=setup,
    #         rounds=ROUNDS,
    #         iterations=ITERATIONS
    #     )

    def test_det(self, benchmark, pkg):
        arr = generate_arrays(pkg, 1)[0]
        setup = lambda: ([arr @ arr.T], {})

        result = benchmark.pedantic(
            target=pkg.linalg.det,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )
    
    def test_norm(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.linalg.norm,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )   