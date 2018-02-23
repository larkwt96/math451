#!/usr/bin/env python3

import numpy as np
import math
from methods import pretty_print
from methods import zeroOut

def getB(size):
    """Post: An is set and global. n is set and global."""
    n = size
    B = np.matrix([[getBij(r, c, n) for c in range(1, n+1)] for r in range(1, n+1)])
    return (B,n)

def getBij(r, c, n):
    """Used for generating A not accessing A. To access A, use A[r,c]"""
    return 2 if r==c else 1 if r==c-1 or r==c+1 else 0

def Pi(i, a, b, n):
    """
        P1 A should have a zero at 1,0
        P2P1A should have a zero at 1,0 and 2,1
        ...
        Pi Pi-1...A should have zero at 1,0, ..., i,i-1

        [c -s, s c] [Ai-1,i-1 Ai,i-1]T = [r 0]T
        where r = l2 norm of A vector
        Rewrite the matrix in terms of c and s instead of A components.
        Now 
        [c s]T = [a/r -b/r]T

        Add that to the matrix, and viola

        Args:
            i: a 1 based index of where the fancy rotation stuff is. Top left is i = 1.
            a: the diagonal element
            b: the element below the diagonal (the one that should be zero)
            n: the size of P
    """
    i = i-1 # because i in the program is really zero based
    r = math.sqrt(a**2 + b**2)
    c = a/r
    s = -b/r
    ret = np.identity(n)
    ret[i,i] = c
    ret[i,i+1] = -s
    ret[i+1,i] = s
    ret[i+1,i+1] = c
    return ret

def qrDecomp(A):
    """
    A1 = A
    A2 = P1T P1 A
    A3 = P2T P1T P1 P2 A
    A4 = (P1T P2T P3T) (P3 P2 P1 A)
    ...
    An = QR

    Finding Pi, see P
    """
    n = len(A)
    Q = np.identity(n)
    R = A
    for i in range(1,n):
        P = Pi(i, R[i-1, i-1], R[i, i-1], n)
        R = P*R
        Q = np.matmul(Q,P.T)
    for r in range(n):
        for c in range(n):
            if abs(Q[r,c]) < 1e-10:
                Q[r,c] = 0
            if abs(R[r,c]) < 1e-10:
                R[r,c] = 0
    return (Q,R)

def doTest(i):
    print('Performing QR decomp for $B_{}$'.format(i), end=' \\\\\n')
    B,_ = getB(i)
    print('$B=$', end=' \\\\\n')
    pretty_print(B)
    Q,R = qrDecomp(B)
    print('$Q=$', end=' \\\\\n')
    pretty_print(Q)
    print('$R=$', end=' \\\\\n')
    pretty_print(R)
    print('Note that it\'s triangular', end=' \\\\\n')
    print('QR=', end=' \\\\\n')
    pretty_print(np.matmul(Q,R))
    print('Note that it\'s $B$ again', end=' \\\\\n')
    print('$QQ^T=$', end=' \\\\\n')
    pretty_print(np.matmul(Q,Q.T))
    #q,r = np.linalg.qr(B)
    #print('q')
    #print(q)
    #print('r')
    #print(r)

def test():
    pass

if __name__ == '__main__':
    for i in range(2, 9):
        doTest(i)
