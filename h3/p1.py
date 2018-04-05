#!/usr/bin/env python3

from res import *
import matplotlib
import matplotlib.pyplot as plt

def p1():
    print('Problem 1')

    print('(a)')
    p1_a()
    print('-~*~-'*2)

    print('(b)')
    p1_b()
    print('-~*~-'*2)

    print('(c)')
    p1_c()
    print('-~*~-'*2)
    print('Note that because of the random initial value, the value for k'
            ' in (b) and the graph in (c) vary from run to run.')

    print('(d)')
    p1_d()
    print('-~*~-'*2)

def p1_a():
    for n in [10, 20, 50, 100]:
        print('Output for A_{}'.format(n))
        A = getA(n)
        print('After 100 iterations...')
        v = getRndV(n)
        print('With random initial x(0)')
        for i in range(100):
            v = iteratePowerMethod(A, v)
            if i % 10 == 0:
                vPrev = v[:]
                v = iteratePowerMethod(A, v, norm=False)
                mu = ampFactor(v, vPrev)
                print('Amplification factor at iteration {} = {}'
                        .format(i, mu))

        vPrev = v[:]
        v = iteratePowerMethod(A, v, norm=False)
        mu = ampFactor(v, vPrev)
        print('Amplification factor = {}'.format(mu))

def p1_b():
    n = 100
    A = getA(n)
    v = getRndV(n)
    ks = [k for k in range(1000)]
    global mus
    mus = [0 for _ in range(1000)]
    for k in range(1000):
        vNext = iteratePowerMethod(A, v, norm=False)
        mu = ampFactor(vNext, v)
        mus[k] = mu
        if k > 0 and abs(mus[k] - mus[k-1]) / abs(mus[k]) > 1e-3:
            lastK = k
        v = normalize(vNext)
    plt.figure(1)
    plt.plot(ks, mus)
    plt.show()
    print('Largest k for which relative error is greater than 1/1000:')
    print('k = {}'.format(lastK))
    print('Final amplification factor = {}'.format(mus[-1]))

def p1_c():
    l1 = mus[-1]
    print('The magnitude of the eigenvalue is assumed to be {}'.format(l1))
    global err
    err = np.array([abs(mus[i] - l1) for i in range(len(mus))], dtype=np.float64)
    plt.figure(2)
    plt.semilogy([i for i in range(len(mus))], err)
    plt.show()

def p1_d():
    ratios = [err[k]/err[k-1] for k in range(1, len(err))]
    plt.figure(3)
    plt.plot(range(len(ratios)), ratios)
    plt.plot([0,1000], [1,1], '-r')
    plt.show()
    print('You can see that most C\'s are below 1. Here is one C: {}'.format(ratios[800]))

if __name__ == '__main__':
    p1()
