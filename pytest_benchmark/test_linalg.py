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

PKGS = []
IDS  = []

try:
    import numpy as np
    PKGS.append(np)
    IDS.append("np")
except:
    print("Could not import dpnp package")

try:
    import dpnp
    PKGS.append(dpnp)
    IDS.append("dpnp")
except:
    print("Could not import dpnp package")

try:
    import cupy
    PKGS.append(cupy)
    IDS.append("cupy")
except:
    print("Could not import cupy package")

try:
    import arrayfire
    af = arrayfire

    # benchmark specific backend and device TODO: argc, argv
    #arrayfire.set_backend(arrayfire.BackendType.oneapi)
    #arrayfire.set_device(0)
    #arrayfire.info()

    PKGS.append(arrayfire)
    IDS.append("arrayfire")
except:
    print("Could not import arrayfire package")

print("imported [" + ", ".join(IDS) + "] packages for benchmarking")

ROUNDS = 10
ITERATIONS = 1

NSIZE = 2**11 # Array column size
NTSIZE = 2**4 # Tensor column size

DTYPE = "float32"

def generate_arrays(pkg, count):
    arr_list = []
    pkg = pkg.__name__
    if "cupy" == pkg:
        for i in range(count):
            #arr_list.append(cupy.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))
            arr_list.append(cupy.random.normal(size=NSIZE * NSIZE).reshape((NSIZE, NSIZE)).astype(DTYPE))
        cupy.cuda.runtime.deviceSynchronize()
    elif "dpnp" == pkg:
        for i in range(count):
            arr_list.append(dpnp.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))
    elif "numpy" == pkg:
        for i in range(count):
            arr_list.append(np.random.normal(size=NSIZE * NSIZE).reshape((NSIZE, NSIZE)).astype(DTYPE))
    elif "arrayfire" == pkg:
        for i in range(count):
            #arr_list.append(np.arange(0, NSIZE * NSIZE, dtype=DTYPE).reshape((NSIZE, NSIZE)))
            # TODO array-api
            arr_list.append(arrayfire.randu((NSIZE, NSIZE), dtype=arrayfire.f32))

    return arr_list

def generate_tensor(pkg, count):
    arr_list = []
    pkg = pkg.__name__
    if "cupy" == pkg:
        for i in range(count):
            arr_list.append(cupy.arange(0, NTSIZE ** 3, dtype=DTYPE).reshape((NTSIZE, NTSIZE, NTSIZE)))
        cupy.cuda.runtime.deviceSynchronize()
    elif "dpnp" == pkg:
        for i in range(count):
            arr_list.append(dpnp.arange(0, NTSIZE ** 3, dtype=DTYPE).reshape((NTSIZE, NTSIZE, NTSIZE)))
    elif "numpy" == pkg:
        for i in range(count):
            arr_list.append(np.arange(0, NTSIZE ** 3, dtype=DTYPE).reshape((NTSIZE, NTSIZE, NTSIZE)))
    elif "arrayfire" == pkg:
        for i in range(count):
            #arr_list.append(np.arange(0, NTSIZE ** 3, dtype=DTYPE).reshape((NTSIZE, NTSIZE, NTSIZE)))
            #TODO array api
            arr_list.append(arrayfire.moddims(arrayfire.range((NTSIZE ** 3), dtype=DTYPE), (NTSIZE, NTSIZE, NTSIZE)))

    return arr_list

@pytest.mark.parametrize(
    "pkg", PKGS, ids=IDS
)
class Eindot:
    def test_dot_a_b(self, benchmark, pkg):
        print(backend)
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
#   def test_cholesky(self, benchmark, pkg):
#       arr = generate_arrays(pkg, 1)[0]
#       setup = lambda: ([pkg.matmul(arr, arr.T)] if pkg == arrayfire else [arr @ arr.T], {})

#       result = benchmark.pedantic(
#           target= pkg.cholesky if pkg == arrayfire else pkg.linalg.cholesky,
#           setup=setup,
#           rounds=ROUNDS,
#           iterations=ITERATIONS)

    def test_svd(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
           target= pkg.svd if pkg == arrayfire else pkg.linalg.svd,
           setup=setup,
           rounds=ROUNDS,
           iterations=ITERATIONS)

    def test_inv(self, benchmark, pkg):
        arr = generate_arrays(pkg, 1)[0]
        setup = lambda: ([pkg.matmul(arr, arr.T)] if pkg == arrayfire else [arr @ arr.T], {})

        result = benchmark.pedantic(
            target=pkg.inverse if pkg == arrayfire else pkg.linalg.inv,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_pinv(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
             target=pkg.pinverse if pkg == arrayfire else pkg.linalg.pinv,
             setup=setup,
             rounds=ROUNDS,
             iterations=ITERATIONS
        )

    def test_det(self, benchmark, pkg):
        arr = generate_arrays(pkg, 1)[0]
        setup = lambda: ([pkg.matmul(arr, arr.T)] if pkg == arrayfire else [arr @ arr.T], {})

        result = benchmark.pedantic(
            target=pkg.det if pkg == arrayfire else pkg.linalg.det,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_norm(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.norm if pkg == arrayfire else pkg.linalg.norm,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )
