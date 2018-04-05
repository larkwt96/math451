#!/usr/bin/env python3

import numpy as np

def dt(k):
    """dt(1)=2, dt(2)=1, dt(3)=1/2, ..."""
    return np.float64(.5)**(k-2)

def f(t, x):
    return np.float64(.5)*x

def runExplicitMethod(x_arr, t_arr, coeff_arr, dt, f):
    """
    Args:
        x_arr (np.array or list) - 
        t_arr (list) -
        coeff_arr (np.array or list) - 
        dt (func(k)) - function where k is passed and dt for that k is returned
        f - function where t, x are passed and x'(t) is returned
        implicit=False (bool) - Whether to run implicit or explicit
    """
    k = len(x_arr)
    delta_t = dt(k)V

def explicitEulerMethod(x0, f, dt, n):
    t_arr = np.array([0], dtype=np.float64)
    for i in range(1, n):
        t_arr.append(t_arr[i-1]+dt(i))

    x_arr=np.array([x0], dtype=np.float64)

    # x_k = x_k-1 + dt*f(t_k-1, x_k-1)
    while len(x_arr) < len(t_arr):
        k = len(x_arr)
        dt = dt(k)
        f = f(t_arr[k-1], x_arr[k-1])
        x_arr.append(np.float64(x_arr[k-1] + dt*f))

def implicitEulerMethod(x0, f, dt, n):
    # x_k = x_k-1 + dt*f(t_k, x_k)
    # x_k = x_k-1 + dt*.5*x_k
    # x_k(1 - dt*.5) = x_k-1
    # x_k = x_k-1/(1-dt*.5)

    t_arr = np.array([0], dtype=np.float64)
    for i in range(1, n):
        t_arr.append(t_arr[i-1]+dt(i))

    x_arr=np.array([x0], dtype=np.float64)

    # x_k = x_k-1/(1-dt*.5)
    while len(x_arr) < len(t_arr):
        k = len(x_arr)
        dt = dt(k)
        x_arr.append(np.float64((x_arr[k-1])/(1-dt*.5))

def implicitCrankNicolsonMethod():
    # x_k = x_k-1 + dt*[1/2*f(t_k, x_k)+1/2*f(t_k-1, x_k-1)]
    # x_k = x_k-1 + dt * .25*x_k+.25*x_k-1
    # x_k(1-dt*.25) = x_k-1 + dt*.25*x_k-1 = x_k-1(1 + dt*.25)
    # x_k = x_k-1 (1+dt*.25)/(1-dt*.25)

    t_arr = np.array([0], dtype=np.float64)
    for i in range(1, n):
        t_arr.append(t_arr[i-1]+dt(i))

    x_arr=np.array([x0], dtype=np.float64)

    # x_k = x_k-1 (1+dt*.25)/(1-dt*.25)
    while len(x_arr) < len(t_arr):
        k = len(x_arr)
        dt = dt(k)
        x_arr.append(x_arr[k-1] * (1+dt*.25) / (1-dt*.25))

def implicitBdf2Method():
    # Bootstrap with implicit CN method
    # x_k = 4/3*x_k-1 - 1/3*x_k-2 + 2/3*dt*f(t_k, x_k)
    # x_k = 4/3*x_k-1 - 1/3*x_k-2 + 2/3*dt*.5*x_k
    # x_k (1 - 1/3*dt) = 4/3*x_k-1 - 1/3*x_k-2
    # x_k = (4/3*x_k-1 - 1/3*x_k-2) / (1 - 1/3*dt)

    t_arr = np.array([0], dtype=np.float64)
    for i in range(1, n):
        t_arr.append(t_arr[i-1]+dt(i))

    x_arr=np.array([x0], dtype=np.float64)
    implicitCrankNicolsonMethod() # TODO

    # x_k = (4/3*x_k-1 - 1/3*x_k-2) / (1 - 1/3*dt)
    while len(x_arr) < len(t_arr):
        k = len(x_arr)
        dt = dt(k)
        xk1 = x_arr[k-1]
        xk2 = x_arr[k-2]
        xk = (4/3*xk1 - 1/3*xk2) / (1-1/3*dt) 
        x_arr.append(xk)
        #TODO
