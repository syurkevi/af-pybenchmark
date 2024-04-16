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
import math

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

MSIZE = 4096 # Array column size
NSIZE = 4096

DTYPE = "float32"
sqrt2 = math.sqrt(2.0)

@pytest.mark.parametrize(
    "pkg", PKGS, ids=IDS
)
class TestBlackScholes:
    def test_black_scholes(self, benchmark, pkg):
        setup = lambda: (generate_arrays(pkg, 5), {})

        result = benchmark.pedantic(
            target=FUNCS[pkg.__name__],
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

def black_scholes_numpy(S, X, R, V, T):
    # S = Underlying stock price
    # X = Strike Price
    # R = Risk free rate of interest
    # V = Volatility
    # T = Time to maturity
    def cnd(x):
        temp = (x > 0)
        erf = lambda arr : np.exp(-arr * arr)
        return temp * (0.5 + erf(x/sqrt2)/2) + (1 - temp) * (0.5 - erf((-x)/sqrt2)/2)

    d1 = np.log(S / X)
    d1 = d1 + (R + (V * V) * 0.5) * T
    d1 = d1 / (V * np.sqrt(T))

    d2 = d1 - (V * np.sqrt(T))
    cnd_d1 = cnd(d1)
    cnd_d2 = cnd(d2)

    C = S * cnd_d1 - (X * np.exp((-R) * T) * cnd_d2)
    P = X * np.exp((-R) * T) * (1 - cnd_d2) - (S * (1 -cnd_d1))

    return (C, P)

def black_scholes_dpnp(S, X, R, V, T):
    def cnd(x):
        temp = (x > 0)
        return temp * (0.5 + dpnp.erf(x/sqrt2)/2) + (1 - temp) * (0.5 - dpnp.erf((-x)/sqrt2)/2)

    d1 = dpnp.log(S / X)
    d1 = d1 + (R + (V * V) * 0.5) * T
    d1 = d1 / (V * dpnp.sqrt(T))

    d2 = d1 - (V * dpnp.sqrt(T))
    cnd_d1 = cnd(d1)
    cnd_d2 = cnd(d2)

    C = S * cnd_d1 - (X * dpnp.exp((-R) * T) * cnd_d2)
    P = X * dpnp.exp((-R) * T) * (1 - cnd_d2) - (S * (1 -cnd_d1))

    return (C, P)

def black_scholes_cupy(S, X, R, V, T):
    def cnd(x):
        temp = (x > 0)
        erf = lambda arr : cupy.exp(-arr * arr)
        return temp * (0.5 + erf(x/sqrt2)/2) + (1 - temp) * (0.5 - erf((-x)/sqrt2)/2)

    d1 = cupy.log(S / X)
    d1 = d1 + (R + (V * V) * 0.5) * T
    d1 = d1 / (V * cupy.sqrt(T))

    d2 = d1 - (V * cupy.sqrt(T))
    cnd_d1 = cnd(d1)
    cnd_d2 = cnd(d2)

    C = S * cnd_d1 - (X * cupy.exp((-R) * T) * cnd_d2)
    P = X * cupy.exp((-R) * T) * (1 - cnd_d2) - (S * (1 -cnd_d1))

    return (C, P)

def black_scholes_arrayfire(S, X, R, V, T):
    def cnd(x):
        temp = (x > 0)
        return temp * (0.5 + af.erf(x/sqrt2)/2) + (1 - temp) * (0.5 - af.erf((-x)/sqrt2)/2)

    d1 = af.log(S / X)
    d1 = d1 + (R + (V * V) * 0.5) * T
    d1 = d1 / (V * af.sqrt(T))

    d2 = d1 - (V * af.sqrt(T))
    cnd_d1 = cnd(d1)
    cnd_d2 = cnd(d2)

    C = S * cnd_d1 - (X * af.exp((-R) * T) * cnd_d2)
    P = X * af.exp((-R) * T) * (1 - cnd_d2) - (S * (1 -cnd_d1))

    af.eval(C, P)
    return (C, P)


def generate_arrays(pkg, count):
    arr_list = []
    pkg = pkg.__name__
    if "cupy" == pkg:
        for i in range(count):
            arr_list.append(cupy.random.rand(MSIZE, NSIZE, dtype=DTYPE))
        cupy.cuda.runtime.deviceSynchronize()
    elif "arrayfire" == pkg:
        for i in range(count):  
            #arr_list.append(af.randu(MSIZE, NSIZE))
            arr_list.append(af.randu((MSIZE, NSIZE)))
    elif "dpnp" == pkg:
        for i in range(count):
            arr_list.append(dpnp.random.rand(MSIZE, NSIZE))
    elif "numpy" == pkg:
        for i in range(count):
            arr_list.append(np.random.rand(MSIZE, NSIZE))

    return arr_list

FUNCS = { "dpnp" : black_scholes_dpnp, "numpy" : black_scholes_numpy, \
         "cupy" : black_scholes_cupy , "arrayfire" : black_scholes_arrayfire}
