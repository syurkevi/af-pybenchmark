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

ROUNDS = 3
ITERATIONS = 100

SAMPLES = 2**21 # Array column size

DTYPE = "float32"

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
    #af.set_seed(1)
    x = af.randu((samples,))
    y = af.randu((samples,))
    result =  4 * af.sum(in_circle(x, y)) / samples
    af.sync()

    return result

def calc_pi_numpy(samples):
    np.random.seed(1)
    x = np.random.rand(samples).astype(np.float32)
    y = np.random.rand(samples).astype(np.float32)
    return 4. * np.sum(in_circle(x, y)) / samples

def calc_pi_cupy(samples):
    #cupy.random.seed(1)
    x = cupy.random.rand(samples, dtype=np.float32)
    y = cupy.random.rand(samples, dtype=np.float32)
    result = 4. * cupy.sum(in_circle(x, y)) / samples
    cupy.cuda.runtime.deviceSynchronize();
    return result

def calc_pi_dpnp(samples):
    dpnp.random.seed(1)
    x = dpnp.random.rand(samples).astype(dpnp.float32)
    y = dpnp.random.rand(samples).astype(dpnp.float32)
    return 4. * dpnp.sum(in_circle(x, y)) / samples

FUNCS = { "dpnp" : calc_pi_dpnp, "numpy" : calc_pi_numpy, \
         "cupy" : calc_pi_cupy , "arrayfire" : calc_pi_af}
