#!/usr/bin/env python3

from res import *
import matplotlib
import matplotlib.pyplot as plt

def p2():
    print('Problem 2')

    n = 100
    A = getA(n)
    v = getRndV(n)
    m1 = -1
    m2 = -1
    m3 = -1
    mappxPrev = -1
    mappx = -1
    mappxVector = []

    for k in range(0, 1000):
        vNext = iteratePowerMethod(A, v, norm=False)

        if not np.isfinite(vNext).all():
            print('Overflow detected breaking at k = {}'.format(k))
            break

        m1 = m2
        m2 = m3
        m3 = ampFactor(vNext, v)
        mappxPrev = mappx
        mappx = m1 - (m2 - m1)**2 / (m3 - 2*m2 + m1)
        mappxVector.append(mappx)
        if k > 0 and abs(mappxPrev - mappx) / abs(mappxPrev) > 1e-3:
            lastK = k

        v = vNext

    print('Largest k for which relative error of the approximated mu values is greater than 1/1000:')
    print('k = {}'.format(lastK))
    print('Final amplification factor = {}'.format(mappx))

    l1 = mappxVector[-1]
    plt.figure(20)
    plt.semilogy(range(len(mappxVector)), np.array([abs(i-l1) for i in mappxVector], dtype=np.float64))
    plt.show()

    print('I\'m guessing the k value is large due to floating point inaccuracies. From the graph, compared to 1.c., it seems to converge slightly faster.')

if __name__ == '__main__':
    p2()
