#!/usr/bin/env python3

from fourthOrderRungeKutta import step
import numpy as np

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
    x0' = x1
    x1' = F(t)/m(t)

    Args:
        t - Time value
        x - The vector of x
    """
    x0 = x[1]
    x1 = F(t, x[0])/m(t, x[0])
    return np.array([x0, x1], dtype=np.float64)

def run(steps=None, dt=None):
    tf = 36000
    d0 = 6371000
    v0 = 0
    x0 = np.array([d0, v0], dtype=np.float64)

    if dt:
        dt = np.float64(dt)
        n = int(tf/dt) + 1
    else:
        dt = np.float64(tf)/steps
        n = steps
    tk1 = 0
    tk = tk1
    xk1 = x0
    xk = xk1
    for k in range(1, n):
        tk = k*dt
        xk = step(f, dt, tk1, xk1)

        tk1 = tk
        xk1 = xk
    if dt:
        newdt = tf - tk1
        tk = tk1 + newdt
        xk = step(f, newdt, tk1, xk1)
    return (dt, tk, xk)

def report(steps1, steps2):
    (dt1, tf1, df1) = run(steps1)
    (dt2, tf2, df2) = run(steps2)
    err = df1[0] - df2[0]
    rel_err = err / df2[0]
    dt = dt2
    df = df2[0]

    print('*'*10)
    print('dt1: {} tf1: {} df1: {}'.format(dt1, tf1, df1[0]))
    print('dt2: {} tf2: {} df2: {}'.format(dt2, tf2, df2[0]))
    print("delta t: {}".format(dt))
    print("final distance: {}".format(df))
    print("error: {}".format(err))
    print("relative error: {}".format(rel_err))

if __name__ == '__main__':
    tf = 36000
    steps = tf

    for k in range(3):
        steps1 = int(steps*(2**(2*k)))
        steps2 = steps1*2
        report(steps1, steps2)
