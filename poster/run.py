#!/usr/bin/env python3

from fourthOrderRungeKutta import step
import numpy as np

def f(t, x):
    a = 10
    C = 400

    res = np.array(x, dtype=np.float64)
    n = len(res)
    for i in range(n):
        res[i] = a*x[i]*(1-x[i]/C)
    return res

def run(steps, tf, x0):
    # Setup
    dt = np.float64(tf)
    tk1 = 0
    tk = tk1
    xk1 = x0
    xk = xk1

    for k in range(1, steps):
        tk = k*dt
        xk = step(f, dt, tk1, xk1)

        tk1 = tk
        xk1 = xk
    return (dt, tk, xk)

if __name__ == '__main__':
    tf = 36000
    steps = tf*2
    n = 2
    x0 = np.array([10 for i in range(n)], dtype=np.float64)
    
    dt, tk, xk = run(steps, tf, x0)
    print(xk)
