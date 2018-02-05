#!/usr/bin/env python3

from math import sqrt # for def of A
from numpy.linalg import det as det_act # for testing
from pandas import DataFrame # for viewing
import time

debug=True

# -~*~- Problem 1 -~*~-
def A(i, j):
    """Returns i-th row and j-th col of A"""
    return sqrt(i+1 + j+1 - 1)

def genA(n):
    """Returns matrix An"""
    return [[A(r,c) for c in range(n)] for r in range(n)]

def det(A):
    """Returns the determinant of matrix A"""
    # exit case
    if len(A) == 1:
        """ |00| """
        return A[0][0]

    # general case
    ret = 0
    # for each row
    for i in range(len(A)):
        # get sign
        sign = 1 if i%2==0 else -1
        # get first element in row
        coeff = A[i][0]
        # calculate the sub matrix
        Asub = [r[1:] for r in (A[0:i]+A[i+1:])]
        # recursive call, depth first
        ret += sign*coeff*det(Asub)
    return ret

def run(n):
    """Returns tuple with (det(An), actual det(An), time to calculate)"""
    A = genA(n)
    start = time.time()
    res = det(A)
    end = time.time()
    res_act = det_act(A)
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
        det,det_act,s = run(n)
        print('n={} in {}s: det(A)={} (actual: {}; err: {}; rel err: {})'.format(n, s, det, det_act, det_act-det, (det_act-det)/det_act))
        n+=1

def test():
    """Testing function"""
    A = [[(sqrt(i+j+2) if i>=j else 0) for j in range(8)] for i in range (8)]
    val = det(A)
    val_act = det_act(A)
    n,s=-1,-1
    print('n={} in {}s: det(A)={} (actual: {}; err: {}; rel err: {})'.format(n, s, val, val_act, val_act-val, (val_act-val)/val_act))

if __name__ == '__main__':
    report(60)
    #test()
