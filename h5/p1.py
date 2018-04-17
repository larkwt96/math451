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

def run(steps, tf, x0):
    dt = np.float64(tf)/steps
    n = steps + 1
    t_arr = np.array([i*dt for i in range(n)], dtype=np.float64)
    x_arr = [None for _ in range(n)]
    x_arr[0] = x0
    for k in range(1, n):
        x_arr[k] = step(f, dt, t_arr[k-1], x_arr[k-1])
    return (t_arr, x_arr)

if __name__ == '__main__':
    tf = 36000
    d0 = 6371000
    v0 = 0
    steps = tf*2
    x0 = np.array([d0, v0], dtype=np.float64)
    (t_arr, x_arr) = run(steps, tf, x0)
    df = x_arr[-1][0]
    err = (x_arr[-1][0] - x_arr[-2][0])/x_arr[-2][0]
    print("df = {}\nrelative err = {}".format(df, err))
