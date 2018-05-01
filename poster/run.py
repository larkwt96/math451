#!/usr/bin/env python3

from fourthOrderRungeKutta import step
import numpy as np
from math import sqrt,ceil

def initloc(n):
    """
    Initialize the locations of the organisms. Draws them on a grid in a
    square.
    """
    global locs
    global rows
    global cols
    rows = int(sqrt(n))
    cols = ceil(n / rows)
    locs = [(int(i/cols), i%cols) for i in range(n)]

def loc(i):
    """Uses a global variable for efficient lookup"""
    global locs
    return locs[i]

def dist(a, b):
    """
    Returns the square of the distance between a and b, where a and b are
    the indexes of x. The location is defined by loc(a) and loc(b).
    
    Doesn't use table lookup since it's not that expensive.
    """
    a_pos = loc(a)
    b_pos = loc(b)
    dx = a_pos[0] - b_pos[0]
    dy = a_pos[1] - b_pos[1]
    return dx*dx + dy*dy

def I(t, x, m):
    n = len(x)
    ret = np.array([0 for _ in range(n)], dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ret[i] += E(t, x[j], m)/dist(i, j)
    print(ret)
    return ret

def E(t, x, m):
    return m*x

def P(t, x, C):
    return x*(1-x/C)

def f(t, x):
    # Weights
    p = 1 # Population rate
    i = 1 # Immigration
    e = 1 # Emigration

    # Parameters
    C = 400 # Carrying capacity
    m = 0.1 # Migration proportion

    # Calculation
    ret = p*P(t, x, C) + i*I(t, x, m) - e*E(t, x, m)
    return ret

def run(steps, tf, x0):
    # Setup
    dt = np.float64(tf)/steps
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
    tf = 10
    steps = tf*2
    n = 7
    initloc(n)
    x0 = np.array([0 for i in range(n)], dtype=np.float64)
    x0[5] = 10
    
    dt, tk, xk = run(steps, tf, x0)
    print(xk)
