#!/usr/bin/env python3

from math import sqrt
from numpy.linalg import det as rdet # for testing
import matplotlib.pyplot as plt
import numpy as np
import time

# λ for copy purposes

# -~*~- Problem 2 -~*~-
def A(i, j):
    """Returns i-th row and j-th col of A"""
    return sqrt(i+1 + j+1 - 1)

def genA(n):
    """Returns matrix An"""
    return [[A(r,c) for c in range(n)] for r in range(n)]

def det(A):
    """Returns the determinant of matrix A. Mutates A."""
    A = np.matrix(A)
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

def pn(A, lam):
    """returns det(A-λ*I)"""
    return det(A-lam*np.eye(len(A)))

def run(n):
    """Returns tuple with (x, det(A-λI)). Tuned for n=2..8 and A above."""
    A = genA(n)
    step = 1000
    if n == 2: start, end = -.5, 3.5
    elif n == 3: start, end = -1.2, 5.5
    elif n == 4: start, end = -.9,8.5
    elif n == 5: start, end = -1.3,12
    elif n == 6: start, end = -1.5,15
    elif n == 7: start, end = -1, .4
    elif n == 8: start, end = -1.8,23 
    else: start, end = -5, 5

    x = np.arange(start, end, (end-start)/step)
    return (x, [pn(A, l) for l in x])

def report():
    """
    Runs the report to be turned in.

    For n=2,...,8
        draw pn and find zeros, eigen values
    """
    plt.figure(figsize=(12,7))
    for n in range(2, 9):
        x,l = run(n)
        plt.subplot(2, 4, n-1)
        plt.title('n='+str(n))
        plt.plot(x, l, x, [0 for i in x])
        plt.gca().set_ylim([-1, 1])
    plt.show()

def test():
    """"""
    pass

def test_plot():
    """Guide to plotting..."""
    def test_f(x):
        return np.sin(x)
    A = genA(3)
    x = np.arange(0.0, 5.0, 0.05)
    plt.figure(1)
    plt.subplot(231)
    plt.plot(x, test_f(x), 'r-')
    plt.subplot(233)
    plt.plot(x, test_f(x), 'b-')
    plt.subplot(232)
    plt.plot(x, [pn(A, l) for l in x], 'g-')
    plt.show()

if __name__ == '__main__':
    report()
    #test()
