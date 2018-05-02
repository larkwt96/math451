#!/usr/bin/env python3

from fourthOrderRungeKutta import step
from secondOrderRungeKutta import step as step2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from math import sqrt,ceil
import time

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

def I(t, x, e):
    n = len(x)
    ret = np.array([0 for _ in range(n)], dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ret[i] += E(t, x[j])/dist(i, j)
    return e*ret

def D(t, x):
    return x

def E(t, x):
    return x

def P(t, x, C):
    return x*(1-x/C)

def f(t, x):
    # Parameters
    p = 1.5 # Population rate
    m = 1 # Immigration rate
    e = .1 # Emigration rate
    C = 400 # Carrying capacity

    # Calculation
    ret = p*P(t, x, C) + m*I(t, x, e) - e*E(t, x)
    return ret

def run(steps, tf, t0, x0):
    # Setup
    dt = np.float64(tf-t0)/steps
    tk1 = t0
    tk = tk1
    xk1 = x0
    xk = xk1

    for k in range(1, steps):
        tk, xk = step(f, dt, tk1, xk1)
        tk1 = tk
        xk1 = xk
    return (dt, tk, xk)

def run2(steps, tf, t0, x0):
    # Setup
    dt = np.float64(tf-t0)/steps
    tk1 = t0
    tk = tk1
    xk1 = x0
    xk = xk1

    for k in range(1, steps):
        tk, xk = step2(f, dt, tk1, xk1)
        tk1 = tk
        xk1 = xk
    return (dt, tk, xk)

def visualize(timeStep, dtPerStep, graphs, x0):
    steps = graphs - 1
    n = steps+1
    t_arr = np.array([0 for i in range(n)], dtype=np.float64)
    x_arr = np.array([0 for i in range(n)], dtype=type(t_arr))

    t_arr[0] = 0
    x_arr[0] = x0
    for i in range(1, n):
        dt, t_arr[i], x_arr[i] = run(
                dtPerStep,
                t_arr[i-1]+timeStep,
                t_arr[i-1],
                x_arr[i-1])
        
    fig = plt.figure()
    for sub in range(n):
        global rows
        global cols
        global locs

        x = range(rows)
        y = range(cols)
        X, Y = np.meshgrid(x, y)
        Z = np.reshape(np.array(x_arr[sub]), (rows, cols))

        ax = fig.add_subplot(1, n, sub+1, projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_title('t={0:.2f}'.format(t_arr[sub]))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Number of Beetles');
    plt.show()
    return dt, t_arr, x_arr

def tableAnalysis():
    tf = 10
    steps = tf*2
    n = 7
    x0 = np.array([0 for i in range(n)], dtype=np.float64)
    x0[5] = 10

    dt, tk, xk = run(steps, tf, 0, x0)
    print(xk)

def visAnalysis():
    n = 100
    initloc(n)
    x0 = np.array([0 for i in range(n)], dtype=np.float64)
    x0[11] = 5
    dt, t_arr, x_arr = visualize(1.5, 100, 5, x0)
    dt, t_arr, x_arr = visualize(1.5, 100, 5, x0)

def l_inf_norm(x):
    """
    Assumes x is non-negative because I don't want to lookup python's negative
    infinity
    """
    max = -1 # (since at least 1 x is positive (should be))
    for i in x:
        if i > max:
            max = i
    return max

def errAnalysis1():
    n = 16
    initloc(n)
    x0 = np.array([0 for i in range(n)], dtype=np.float64)
    x0[8] = 5
    steps_0 = 6*100
    tf = 6
    size = 5
    dt = [0 for i in range(size)]
    xf = [0 for i in range(size)]
    err = [0 for i in range(size)]
    calc_time = [0 for i in range(size)]
    steps = [steps_0*(2**i) for i in range(size)]
    for i in range(size):
        start = time.time()
        dt[i], _, xf[i] = run(steps[i], tf, 0, x0)
        end = time.time()
        calc_time[i] = end - start

    print('Starts with {} beetles in tree {} to t={}'.format(x0[8], 8, tf))
    for i in range(size):
        print('Run {}: dt={} xf={} steps={} time={}'.format(i, dt[i], xf[i], steps[i], calc_time[i]))
    for i in range(1, size):
        err[i] = abs(xf[i] - xf[i-1])
        print('Error {}: xf_{}-xf_{}={}'.format(i, i, i-1, l_inf_norm(err[i])))

def errAnalysis2():
    n = 16
    initloc(n)
    x0 = np.array([0 for i in range(n)], dtype=np.float64)
    x0[8] = 5
    tf = 6
    size = 5

    steps = 6*100*(2**5)
    start = time.time()
    dt, _, xf4 = run(steps, tf, 0, x0)
    end = time.time()
    dt, _, xf2 = run2(steps, tf, 0, x0)
    calc_time4 = end - start

    print('Starts with {} beetles in tree {} to t={} with dt={}'.format(x0[8], 8, tf, dt))
    print('Second order: xf={} steps={} time={}'.format(xf2, steps, calc_time2))
    print('Fourth order: xf={} steps={} time={}'.format(xf4, steps, calc_time4))
    print('err = {}'.format(xf4-xf2))
    print('l_inf_norm of that = {}'.format(l_inf_norm(xf4-xf2)))
    print('time diff = {}'.format(calc_time4-calc_time2))

if __name__ == '__main__':
    errAnalysis2()
