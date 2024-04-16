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

ROUNDS = 30
ITERATIONS = 1

NSIZE = 2**9 # Array column size

DTYPE = "float32"

def generate_arrays(pkg, count):
    arr_list = []
    pkg = pkg.__name__
    if "cupy" == pkg:
        for i in range(count):
            arr_list.append(cupy.arange(1, NSIZE * NSIZE + 1, dtype=DTYPE).reshape((NSIZE, NSIZE)))
        cupy.cuda.runtime.deviceSynchronize()
    # elif "arrayfire" == pkg:
    #     for i in range(count):  
    #         arr_list.append(arrayfire.arange(1, NSIZE * NSIZE + 1, dtype=DTYPE).reshape((NSIZE, NSIZE)) / NSIZE)
    elif "dpnp" == pkg:
        for i in range(count):
            arr_list.append(dpnp.arange(1, NSIZE * NSIZE + 1, dtype=DTYPE).reshape((NSIZE, NSIZE)))
    elif "numpy" == pkg:
        for i in range(count):
            arr_list.append(np.arange(1, NSIZE * NSIZE + 1, dtype=DTYPE).reshape((NSIZE, NSIZE)))
    elif "arrayfire" == pkg:
        for i in range(count):
            #arr_list.append(np.arange(1, NSIZE * NSIZE + 1, dtype=DTYPE).reshape((NSIZE, NSIZE)))
            arr_list.append(arrayfire.moddims(arrayfire.range((NSIZE * NSIZE,), dtype=arrayfire.f32)+1, (NSIZE, NSIZE)))

    return arr_list

@pytest.mark.parametrize(
    "pkg", PKGS, ids=IDS
)
class TestElementwise:
    # cumulative elementwise benchmark
    def test_group_elementwise(self, benchmark, pkg):
        setup = lambda: ([generate_arrays(pkg, 1)[0] / (NSIZE * NSIZE)], {})

        def func(arr):
            if pkg == arrayfire:
                res = pkg.exp(pkg.cos(pkg.asinh(arr))) +\
                    pkg.cbrt(pkg.log(arr) * pkg.expm1(-pkg.sqrt(arr)))
                arrayfire.eval(res)
            else:
                res = pkg.exp(pkg.cos(pkg.arcsinh(arr))) +\
                    pkg.cbrt(pkg.log(arr) * pkg.expm1(-pkg.sqrt(arr)))


        result = benchmark.pedantic(
            target=func,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

'''
# individual elementwise benchmarks
    def test_arccos(self, benchmark, pkg):
        setup = lambda: ([generate_arrays(pkg, 1)[0] / (NSIZE * NSIZE)], {})

        result = benchmark.pedantic(
            target=pkg.acos if pkg == arrayfire else pkg.arccos,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_arccosh(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.acosh if pkg == arrayfire else pkg.arccosh,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_arcsin(self, benchmark, pkg):
        setup = lambda: ([generate_arrays(pkg, 1)[0] / (NSIZE * NSIZE)], {})

        result = benchmark.pedantic(
            target=pkg.asin if pkg == arrayfire else pkg.arcsin,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_arcsinh(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.asinh if pkg == arrayfire else pkg.arcsinh,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_arctan(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.atan if pkg == arrayfire else pkg.arctan,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_arctanh(self, benchmark, pkg):
        setup = lambda: ([(generate_arrays(pkg, 1)[0] - 1) / (NSIZE * NSIZE)], {})

        result = benchmark.pedantic(
            target=pkg.atanh if pkg == arrayfire else pkg.arctanh,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_cbrt(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.cbrt,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_cos(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.cos,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_cosh(self, benchmark, pkg):
        setup = lambda: ([generate_arrays(pkg, 1)[0] / (NSIZE * NSIZE)], {})

        result = benchmark.pedantic(
            target=pkg.cosh,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_sin(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.sin,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_sinh(self, benchmark, pkg):
        setup = lambda: ([generate_arrays(pkg, 1)[0] / (NSIZE * NSIZE)], {})

        result = benchmark.pedantic(
            target=pkg.sinh,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

#   def test_degrees(self, benchmark, pkg):
#       setup = lambda: (generate_arrays(pkg, 1), {})

#       result = benchmark.pedantic(
#           target=pkg.degrees,
#           setup=setup,
#           rounds=ROUNDS,
#           iterations=ITERATIONS
#       )

    def test_exp(self, benchmark, pkg):
        setup = lambda: ([generate_arrays(pkg, 1)[0] / (NSIZE * NSIZE)], {})

        result = benchmark.pedantic(
            target=pkg.exp,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

#   def test_exp2(self, benchmark, pkg):
#       setup = lambda: ([generate_arrays(pkg, 1)[0] / (NSIZE * NSIZE)], {})

#       result = benchmark.pedantic(
#           target=pkg.exp2,
#           setup=setup,
#           rounds=ROUNDS,
#           iterations=ITERATIONS
#       )

    def test_expm1(self, benchmark, pkg):
        setup = lambda: ([generate_arrays(pkg, 1)[0] / (NSIZE * NSIZE)], {})

        result = benchmark.pedantic(
            target=pkg.expm1,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_log(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.log,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_log10(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.log10,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_log1p(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.log1p,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_sqrt(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.sqrt,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

#   def test_square(self, benchmark, pkg):
#       setup = lambda: (generate_arrays(pkg, 1), {})

#       result = benchmark.pedantic(
#           target=pkg.square,
#           setup=setup,
#           rounds=ROUNDS,
#           iterations=ITERATIONS
#       )

    def test_tan(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.tan,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    def test_tanh(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 1), {})

        result = benchmark.pedantic(
            target=pkg.tanh,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )
'''

#   def test_reciprocal(self, benchmark, pkg):
#       setup = lambda: (generate_arrays(pkg, 1), {})

#       result = benchmark.pedantic(
#           target=pkg.reciprocal,
#           setup=setup,
#           rounds=ROUNDS,
#           iterations=ITERATIONS
#       )

#def generate_arrays(pkg, count):
#    arr_list = []
#    pkg = pkg.__name__
#    if "cupy" == pkg:
#        for i in range(count):
#            arr_list.append(cupy.arange(1, NSIZE * NSIZE + 1, dtype=DTYPE).reshape((NSIZE, NSIZE)))
#        cupy.cuda.runtime.deviceSynchronize()
#    elif "arrayfire" == pkg:
#        af.device_gc()
#        for i in range(count):  
#            arr_list.append(af.arange(1, NSIZE * NSIZE + 1, dtype=DTYPE).reshape((NSIZE, NSIZE)))
#    elif "dpnp" == pkg:
#        for i in range(count):
#            arr_list.append(dpnp.arange(1, NSIZE * NSIZE + 1, dtype=DTYPE).reshape((NSIZE, NSIZE)))
#    elif "numpy" == pkg:
#        for i in range(count):
#            arr_list.append(np.arange(1, NSIZE * NSIZE + 1, dtype=DTYPE).reshape((NSIZE, NSIZE)))
#
#    return arr_list
