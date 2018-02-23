#!/usr/bin/env python3

import p1
import p2
from methods import pretty_print
import numpy as np

def getA(n):
    """Post: An is set and global. n is set and global."""
    return np.matrix([[getAij(r, c, n) for c in range(1, n+1)] for r in range(1, n+1)])

def getAij(r, c, n):
    """Used for generating A not accessing A. To access A, use A[r,c]"""
    return (n+1)/n if r==c else 1/n


def err(row, i):
    return sum(row) - row[i]

def run(B, i):
    for i in range(1,10):
        Q,R = p2.qrDecomp(B)
        B = np.matmul(R,Q)
        print('$B$ after $'+str(i)+'^th$ iteration \\\\')
        pretty_print(B)
    print('Eigen values are \\\\')
    print('$\\lambda = $')
    for i in range(n):
        print('${}$ with err ${}$,\\\\'.format(B[i,i], err(B[i], i)))
    eigs,_ = np.linalg.eig(p2.getB(i)[0])
    print('actual eigen values: ${}$\\\\'.format(eigs))




if __name__ == '__main__':
    for i in range(2, 8):
        B,n = p2.getB(i)
        run(B)
    for i in range(2, 3):
