#!/usr/bin/env python3

import numpy as np

def setA(n):
    """Post: An is set and global. n is set and global."""
    global A
    globals()['n'] = n
    A = np.matrix([[getAij(r, c, n) for c in range(1, n+1)] for r in range(1, n+1)])

def getAij(r, c, n):
    """Used for generating A not accessing A. To access A, use A[r,c]"""
    return (n+1)/n if r==c else 1/n

def delta(r,c):
    """Iij"""
    return 1 if i == j else 0

def setNextA():
    # iterate for every column, getP, get W, calculate next A
    # use np.matmul
    pass # TODO

def getP(w):
    """I - 2 wwT"""
    return [[delta(r,c)-2*w[r]*w[c] for c in range(1,n+1)] for r in range(1,n+1)]

def getW(Acol):
    """
    Returns w for a given A's first col.
    """
    w = np.zeros(len(Acol))
    alpha = getAlpha(Acol)
    r = getR(alpha)
    w[0] = 0
    w[1] = (Acol[1]-alpha)/2/r
    for i in range(2, len(w)):
        w[i] = Acol[i]/2/r

def getAlpha(Acol):
    """
    Gets alpha used in calculating w from Acol.

    Returns:
        alpha = -sign(A_21)*(sum j=2 to n [a_j1^2])^.5
    """
    pass # TODO

def getR(a):
    """
    Args:
        a: alpha
    """
    pass # TODO

if __name__ == '__main__':
    setA(4)
    for i in range(1, 5):
        setA(i)
        print(i*A)
