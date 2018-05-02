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
    global p
    global m
    global e
    global C
    global custF

    # Parameters
    if not custF:
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

def visualize(timeStep, dtPerStep, graphs, x0, sideways=False):
    steps = graphs - 1
    n = steps+1
    t_arr = np.array([0 for i in range(n)], dtype=np.float64)
    x_arr = np.array([0 for i in range(n)], dtype=type(t_arr))

    t_arr[0] = 0
    x_arr[0] = x0
    for i in range(1, n):
        dt, t_arr[i], x_arr[i] = run2(
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

        if sideways:
            ax = fig.add_subplot(n, 1, sub+1, projection='3d')
        else:
            ax = fig.add_subplot(1, n, sub+1, projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_title('t={0:.2f} with dt={1}'.format(t_arr[sub], dt))
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
    """
    This error analysis is error of rk4
    """
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
    """
    This is error analysis of rk2 with rk4 as a baseline
    """
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
    calc_time4 = end-start

    start = time.time()
    dt, _, xf2 = run2(steps, tf, 0, x0)
    end = time.time()
    calc_time2 = end - start

    print('Starts with {} beetles in tree {} to t={} with dt={}'.format(x0[8], 8, tf, dt))
    print('Second order: xf={} steps={} time={}'.format(xf2, steps, calc_time2))
    print('Fourth order: xf={} steps={} time={}'.format(xf4, steps, calc_time4))
    print('err = {}'.format(xf4-xf2))
    print('l_inf_norm of that = {}'.format(l_inf_norm(xf4-xf2)))
    print('time diff = {}'.format(calc_time4-calc_time2))

def errAnalysis3():
    """
    This is an error analysis of rk2
    """
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
        dt[i], _, xf[i] = run2(steps[i], tf, 0, x0)
        end = time.time()
        calc_time[i] = end - start

    print('Starts with {} beetles in tree {} to t={}'.format(x0[8], 8, tf))
    for i in range(size):
        print('Run {}: dt={} xf={} steps={} time={}'.format(i, dt[i], xf[i], steps[i], calc_time[i]))
    for i in range(1, size):
        err[i] = abs(xf[i] - xf[i-1])
        print('Error {}: xf_{}-xf_{}={}'.format(i, i, i-1, l_inf_norm(err[i])))

def timeGraph():
    """dt vs time graph"""
    size = 5
    tf = 6
    steps = [2*100*(2**i) for i in range(size)]
    dt = [tf/steps[i] for i in range(size)]
    print(dt)
    rk4 = [1.706394910812378, 3.397844076156616, 6.788722038269043, 13.825963020324707, 27.090019941329956]
    rk2 = [0.33904504776000977, 0.6788828372955322, 1.3788299560546875, 2.779728889465332, 5.521608114242554]
    fig = plt.figure()
    plt.title('Figure 2')
    plt.xlabel('dt')
    plt.ylabel('Computation Time')
    plt.plot(dt, rk2, 'r-', label='RK-2')
    plt.plot(dt, rk4, 'b-', label='RK-4')
    plt.legend()
    plt.show()

def visAnalysis1():
    print('A base for visualization')
    global p
    global m
    global e
    global C
    global custF


    # Parameters
    custF = True
    p = 1.5 # Population rate
    m = .8 # Immigration rate
    e = .1 # Emigration rate
    C = 400 # Carrying capacity

    n = 100
    initloc(n)
    x0 = np.array([0 for i in range(n)], dtype=np.float64)
    x0[11] = 5
    dt, t_arr, x_arr = visualize(1.5, 300, 4, x0)
    custF = False

def visAnalysis2():
    print('Cripple transfer rate')
    global p
    global m
    global e
    global C
    global custF

    # Parameters
    custF = True
    p = 1.5 # Population rate
    m = .08 # Immigration rate
    e = .01 # Emigration rate
    C = 400 # Carrying capacity

    n = 100
    initloc(n)
    x0 = np.array([0 for i in range(n)], dtype=np.float64)
    x0[11] = 5
    dt, t_arr, x_arr = visualize(1.5, 300, 4, x0)
    custF = False

def visAnalysis3():
    print('decrease population rate')
    global p
    global m
    global e
    global C
    global custF

    # Parameters
    custF = True
    p = .5 # Population rate
    m = .8 # Immigration rate
    e = .1 # Emigration rate
    C = 400 # Carrying capacity

    n = 100
    initloc(n)
    x0 = np.array([0 for i in range(n)], dtype=np.float64)
    x0[11] = 5
    dt, t_arr, x_arr = visualize(1.5, 300, 4, x0)
    custF = False

def visAnalysis4():
    print('different starting place')
    global p
    global m
    global e
    global C
    global cuatF

    # Parameters
    custF = True
    p = 1.5 # Population rate
    m = .8 # Immigration rate
    e = .1 # Emigration rate
    C = 400 # Carrying capacity

    n = 100
    initloc(n)
    x0 = np.array([0 for i in range(n)], dtype=np.float64)
    x0[0] = 5
    x0[25] = 5
    x0[50] = 5
    x0[75] = 5
    dt, t_arr, x_arr = visualize(1.5, 300, 4, x0)
    custF = False

if __name__ == '__main__':
    visAnalysis1()
    visAnalysis2()
    visAnalysis3()
    visAnalysis4()
