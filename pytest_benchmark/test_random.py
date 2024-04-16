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

ROUNDS = 10
ITERATIONS = 400

NNUMBERS = 2**20

@pytest.mark.parametrize(
    "pkg", PKGS, ids=IDS
)
class TestRandom:
#   def test_beta(self, benchmark, pkg):
#       result = benchmark.pedantic(
#           target=pkg.random.beta,
#           args=(
#               4.0,
#               5.0,
#               NNUMBERS,
#           ),
#           rounds=ROUNDS,
#           iterations=ITERATIONS,
#       )

#   def test_exponential(self, benchmark, pkg):
#       result = benchmark.pedantic(
#           target=pkg.random.exponential,
#           args=(
#               4.0,
#               NNUMBERS,
#           ),
#           rounds=ROUNDS,
#           iterations=ITERATIONS,
#       )

#   def test_gamma(self, benchmark, pkg):
#       result = benchmark.pedantic(
#           target=pkg.random.gamma,
#           args=(
#               2.0,
#               4.0,
#               NNUMBERS,
#           ),
#           rounds=ROUNDS,
#           iterations=ITERATIONS,
#       )

    def test_normal(self, benchmark, pkg):
        result = benchmark.pedantic(
            target=pkg.randn if pkg == arrayfire else pkg.random.normal,
            args=((NNUMBERS,),) if pkg == arrayfire else (0.0, 1.0, NNUMBERS),
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )

    def test_uniform(self, benchmark, pkg):
        result = benchmark.pedantic(
            target=pkg.randu if pkg == arrayfire else pkg.random.uniform,
            args=((NNUMBERS,),) if pkg == arrayfire else (0.0, 1.0, NNUMBERS),
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )
