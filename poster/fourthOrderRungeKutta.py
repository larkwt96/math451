#!/usr/bin/env python3

def _F1(f, dt, tk1, xk1):
    """
    Args:
        f - A function: f(t, t_k-1, x_k-1) where x_k-1 is a vector of the x_k-1
            values
        dt - Delta t
        tk1 - t_k-1
        xk1 - x_k-1
    """
    return dt * f(tk1, xk1)

def _F2(f, dt, tk1, xk1):
    arg1 = tk1 + dt/2
    arg2 = xk1 + 1/2*_F1(f, dt, tk1, xk1)
    return dt * f(arg1, arg2)

def _F3(f, dt, tk1, xk1):
    arg1 = tk1 + dt/2
    arg2 = xk1 + 1/2*_F2(f, dt, tk1, xk1)
    return dt * f(arg1, arg2)

def _F4(f, dt, tk1, xk1):
    arg1 = tk1 + dt
    arg2 = xk1 + 1*_F3(f, dt, tk1, xk1)
    return dt * f(arg1, arg2)

def step(f, dt, tk1, xk1):
    """
    Get xk from tk-1 and xk-1 given f and dt

    Args:
        f - A function: f(t, t_k-1, x_k-1) where x_k-1 is a vector of the x_k-1
            values
        dt - Delta t
        tk1 - t_k-1
        xk1 - x_k-1
    """
    F1val = _F1(f, dt, tk1, xk1)
    F2val = _F2(f, dt, tk1, xk1)
    F3val = _F3(f, dt, tk1, xk1)
    F4val = _F4(f, dt, tk1, xk1)
    xk = xk1 + 1/6*F1val + 2/6*F2val + 2/6*F3val + 1/6*F4val
    return xk
