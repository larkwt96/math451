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


def err(B, i):
    sum = 0
    for r in range(len(B)):
        if r != i:
            sum += abs(B[i,r])
    return sum

def run(B, i):
    for i in range(1,10):
        Q,R = p2.qrDecomp(B)
        B = np.matmul(R,Q)
        print('$B$ after $'+str(i)+'^th$ iteration \\\\')
        pretty_print(B)
    print('Eigen values are \\\\')
    print('$\\lambda = $')
    for i in range(n):
        print('${}$ with err ${}$,\\\\'.format(B[i,i], err(B, i)))
    eigs,_ = np.linalg.eig(p2.getB(i)[0])
    for i in range(len(eigs)):
        print('actual eigen values: ${}$\\\\'.format(eigs[i]))



import numpy as np
import math
from methods import pretty_print

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
    return 1 if c == r else 0

def doHouseholdersTransform():
    global A
    for c in range(n-2):
        col = A[:,c][c:]
        #print('col')
        #print(col)
        w = getW(col)
        #print('w')
        #print(w)
        P = getP(w)
        #print('P')
        #print(P)
        A = P*A*P
        #print('A')
        #print(A)
    # zero ends
    for r in range(n):
        for c in range(n):
            if abs(A[r,c]) < 1e-10:
                A[r,c] = 0

def getP(w):
    """I - 2 wwT"""
    P = np.identity(n)
    base = n-len(w)
    for c in range(len(w)):
        for r in range(len(w)):
            P[base+r,base+c] = P[base+r,base+c] - 2*w[r]*w[c]
    return P

def getW(Acol):
    """
    Returns w for a given A's first col.
    """
    w = np.zeros(len(Acol))
    alpha = getAlpha(Acol)
    r = getR(Acol, alpha)
    w[0] = 0
    w[1] = (Acol[1]-alpha)/2/r
    for i in range(2, len(w)):
        w[i] = Acol[i]/2/r
    return w

def getAlpha(Acol):
    """
    Gets alpha used in calculating w from Acol.

    Returns:
        alpha = -sign(A_21)*(sum j=2 to n [a_j1^2])^.5
    """
    # note A_21 is A[1] since 2 -> 1 and 1 is taken care of
    n = len(Acol)
    sign = -1 if Acol[1] < 0 else 1
    l2Norm = math.sqrt(sum([Acol[j]**2 for j in range(1, n)]))
    alpha = -sign*l2Norm
    return alpha

def getR(Acol, a):
    """
    r = (.5*a^2 - .5*A_21*a)^.5

    Args:
        Acol: The working column of A
        a: alpha
    """
    return math.sqrt(.5*a**2 - .5*Acol[1]*a)

def isSymmetric():
    points = []
    for r in range(n):
        for c in range(r+1, n):
            if abs(A[r,c]-A[c,r]) > 1e-10:
                points.append((r,c))
    return points

def isTridiagonal():
    points = []
    for r in range(n):
        for c in range(r+2, n):
            if A[r,c] != 0 or A[c,r] != 0:
                points.append((r,c))
    return points

def doTest(n):
    setA(n)
    print('Householders for n={}'.format(n))
    print('Before householders transform')
    pretty_print(A)
    doHouseholdersTransform()
    print('After householders transform')
    pretty_print(A)

    sym = isSymmetric()
    if len(sym) > 0:
        print("Here are points lacking symmetry:")
        for p in sym:
            print('({},{})'.format(p[0],p[1]))
    else:
        print("Matrix is symmetric")

    tri = isTridiagonal()
    if len(tri) > 0:
        print("Here are points violating tridiagonal:")
        for p in tri:
            print('({},{})'.format(p[0],p[1]))
    else:
        print("Matrix is tridiagonal")


if __name__ == '__main__':
    for i in range(2, 8):
        setA(i)
        doHouseholdersTransform()
        run(A, i)
#!/usr/bin/env python3

