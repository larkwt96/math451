#!/usr/bin/env python3

import math
from random import uniform
import numpy as np

def Aij(r, c, n):
    """Return A_rc where A has size n"""
    if r == c:
        return 2
    elif abs(r-c) == 1:
        return -1
    else:
        return 0

def getA(n):
    """Return A of size n"""
    return np.matrix([[Aij(r,c,n) for c in range(n)] for r in range(n)],
            dtype=np.float64)

def lInfNorm(v):
    """L-infinity norm of v"""
    max = -math.inf
    for i in v:
        if abs(i) > max:
            max = abs(i)
    return max

def normalize(v):
    """Return v as unit vector"""
    return v/lInfNorm(v)

def ampFactor(xk, xkPrev):
    """Calculate the amplification factor of x_k and x_k-1"""
    return np.float64(lInfNorm(xk)/lInfNorm(xkPrev))

def getRndV(n, lBound=-1, uBound=1):
    """Return a random initial value for v of size n"""
    return np.array([uniform(lBound, uBound) for _ in range(n)], dtype=np.float64)

def iteratePowerMethod(A, v, norm=True):
    vNext = np.squeeze(np.asarray(A.dot(v)))
    if norm:
        vNext = normalize(vNext)
    return vNext

if __name__ == '__main__':
    print(getA(5))
