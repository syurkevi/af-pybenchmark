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
ITERATIONS = 4

NNUMBERS = 2**16
PKGS = [dpnp, np, cupy, af]
IDS = [pkg.__name__ for pkg in PKGS]

@pytest.mark.parametrize(
    "pkg", PKGS, ids=IDS
)
class TestRandom:
    def test_beta(self, benchmark, pkg):
        result = benchmark.pedantic(
            target=pkg.random.beta,
            args=(
                4.0,
                5.0,
                NNUMBERS,
            ),
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )

    def test_exponential(self, benchmark, pkg):
        result = benchmark.pedantic(
            target=pkg.random.exponential,
            args=(
                4.0,
                NNUMBERS,
            ),
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )

    def test_gamma(self, benchmark, pkg):
        result = benchmark.pedantic(
            target=pkg.random.gamma,
            args=(
                2.0,
                4.0,
                NNUMBERS,
            ),
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )

    def test_normal(self, benchmark, pkg):
        result = benchmark.pedantic(
            target=pkg.random.normal,
            args=(
                0.0,
                1.0,
                NNUMBERS,
            ),
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )

    def test_uniform(self, benchmark, pkg):
        result = benchmark.pedantic(
            target=pkg.random.uniform,
            args=(
                0.0,
                1.0,
                NNUMBERS,
            ),
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )
