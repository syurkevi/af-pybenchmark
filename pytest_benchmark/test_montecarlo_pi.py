#!/usr/bin/python

#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################


import pytest

import arrayfire as af
import numpy as np
import dpnp
import cupy

ROUNDS = 30
ITERATIONS = 1

SAMPLES = 2**8 # Array column size

DTYPE = "float32"
PKGS = [dpnp, np, cupy, af]
IDS = [pkg.__name__ for pkg in PKGS]

@pytest.mark.parametrize(
    "pkg", PKGS, ids=IDS
)
class TestPi:
    def test_pi(self, benchmark, pkg):
        result = benchmark.pedantic(
            target=FUNCS[pkg.__name__],
            rounds=ROUNDS,
            iterations=ITERATIONS,
            args=[SAMPLES]
        )

# Having the function outside is faster than the lambda inside
def in_circle(x, y):
    return (x*x + y*y) < 1

def calc_pi_af(samples):
    af.random.set_seed(1)
    x = af.randu(samples)
    y = af.randu(samples)
    result =  4 * af.sum(in_circle(x, y)) / samples

    af.eval(result)
    af.sync()

    return result

def calc_pi_numpy(samples):
    np.random.seed(1)
    x = np.random.rand(samples).astype(np.float32)
    y = np.random.rand(samples).astype(np.float32)
    return 4. * np.sum(in_circle(x, y)) / samples

def calc_pi_cupy(samples):
    cupy.random.seed(1)
    x = cupy.random.rand(samples, dtype=np.float32)
    y = cupy.random.rand(samples, dtype=np.float32)
    return 4. * cupy.sum(in_circle(x, y)) / samples

def calc_pi_dpnp(samples):
    dpnp.random.seed(1)
    x = dpnp.random.rand(samples).astype(dpnp.float32)
    y = dpnp.random.rand(samples).astype(dpnp.float32)
    return 4. * dpnp.sum(in_circle(x, y)) / samples

FUNCS = { "dpnp" : calc_pi_dpnp, "numpy" : calc_pi_numpy, \
         "cupy" : calc_pi_cupy , "arrayfire" : calc_pi_af}