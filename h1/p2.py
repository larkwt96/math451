#!/usr/bin/env python3

from math import sqrt # for def of A
from numpy.linalg import det as rdet # for testing
import numpy
import time

debug=True

# -~*~- Problem 2 -~*~-
def A(i, j):
    """Returns i-th row and j-th col of A"""
    return sqrt(i+1 + j+1 - 1)

def genA(n):
    """Returns matrix An"""
    return [[A(r,c) for c in range(n)] for r in range(n)]

def det(A):
    """Returns the determinant of matrix A. Mutates A."""
    A = numpy.matrix(A)
    for c in range(len(A)):
        for r in range(c+1, len(A)):
            to_elim = A[r,c]
            diag_factor = A[c,c]
            if diag_factor == 0: continue
            scale = float(to_elim/diag_factor)
            A[r] = (A[r] - scale*A[c])
    prod = 1
    for i in range(len(A)):
        prod *= A[i,i]
    return prod

def run(n):
    """Returns tuple with (det(An), actual det(An), time to calculate)"""
    A = genA(n)
    start = time.time()
    res = det(A)
    end = time.time()
    res_act = rdet(A)
    return (res, res_act, end-start)

def report(max_sec=60):
    """
    Runs the report to be turned in.

    For n=1,2,...
        det(An)
        runtime

    n stops when det takes more than max_sec seconds to run (1 min by default).
    """
    n = 1
    s = 0
    while s <= max_sec:
        det,rdet,s = run(n)
        print('n={} in {}s: det(A)={} (actual: {}; err: {}; rel err: {})'
                .format(n,
                    s,
                    det,
                    rdet,
                    rdet-det,
                    (rdet-det)/rdet))
        n+=1

def test():
    """Testing function"""
    A = [[(sqrt(i+j+2) if i>=j else 0) for j in range(3)] for i in range (3)]
    val = det(A)
    val_act = rdet(A)
    n,s=-1,-1
    print('n={} in {}s: det(A)={} (actual: {}; err: {}; rel err: {})'.format(n, s, val, val_act, val_act-val, (val_act-val)/val_act))

if __name__ == '__main__':
    report(60)
    #test()
