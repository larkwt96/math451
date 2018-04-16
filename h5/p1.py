#!/usr/bin/env python3

from fourthOrderRungeKutta import step

def T(t, d):
    """
    Args:
        t - Time value
        d - d(t)
    """
    if t < 600:
        return 12
    else:
        return 0

def m(t, d):
    """
    Args:
        t - Time value
        d - d(t)
    """
    if t < 600:
        return 1 - 0.9*t/600
    else:
        return 0.1

def G(t, d):
    """
    Args:
        t - Time value
        d - d(t)
    """
    return - ((6371000)**2) * 10 * m(t, d) / (d**2)

def F(t, d):
    """
    Args:
        t - Time value
        d - d(t)
    """
    return T(t, d) + G(t, d)

def f(t, x):
    """
    x1' = x2
    x2' = F(t)/m(t)
    Args:
        t - Time value
        x - The vector of x
    """
    x1 = x[1]
    x2 = F(t, x[0])/m(t, x[0])
    x[0] = x1
    x[1] = x2
    return x

def run(n):
    pass
