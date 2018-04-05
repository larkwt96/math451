#!/usr/bin/env python3

from res import *
import matplotlib
import matplotlib.pyplot as plt

def getA():
    """An override of the method from res"""
    return np.matrix([[2, 0, 0], [0, 2, 0], [0, 0, 1]], np.float64)

def p3():
    print('Problem 3')

    print('(a)')
    p3_a()
    print('-~*~-'*2)

    print('(b)')
    p3_b()
    print('-~*~-'*2)

def p3_a():
    n = 3
    A = getA()

    for i in range(10):
        v = getRndV(n, -10, 10)
        print('Initial v = {}'.format(v))
        for k in range(1000):
            vNext = iteratePowerMethod(A, v, norm=False)
            mu = ampFactor(vNext, v)
            v = normalize(vNext)
        print('After 1000 iterations')
        print('Final amplification factor = {}'.format(mu))
        print('Final v = {}'.format(v))
        print()
    print('It seems that the dominant direction of the initial vector'
            ' selects which eigen value to converge to. For example,'
            ' [7,5,3] converges to [1,0,0] and [5,7,3] converges to'
            ' [0,1,0]')

def p3_b():
    print('I don\'t think so since it depends on the ration of larger eigen values being strictly less than 1, which allows powers of it to converge to 0. The strange part is that the examples seem to converge most of the time.')

if __name__ == '__main__':
    p3()
