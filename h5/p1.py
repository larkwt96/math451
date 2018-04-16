#!/usr/bin/env python3

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

def F1(f, dt, tk1, xk1):
    return dt * f(tk1, xk1)

def F2(f, dt, tk1, xk1):
    arg1 = tk1 + dt/2
    arg2 = xk1 + dt/2*F1(f, dt, tk1, xk1)
    return dt * f(arg1, arg2)

def F3(f, dt, tk1, xk1):
    arg1 = tk1 + dt/2
    arg2 = xk1 + dt/2*F2(f, dt, tk1, xk1)
    return dt * f(arg1, arg2)

def F4(f, dt, tk1, xk1):
    arg1 = tk1 + dt
    arg2 = xk1 + dt*F3(f, dt, tk1, xk1)
    return dt * f(arg1, arg2)

def step(f, dt, tk1, xk1):
    """
    Get xk from tk-1 and xk-1
    """
    F1val = F1(f, dt, tk1, xk1)
    F2val = F2(f, dt, tk1, xk1)
    F3val = F3(f, dt, tk1, xk1)
    F4val = F4(f, dt, tk1, xk1)
    return xk1 + 1/6*F1val + 2/6*F2val + 2/6*F3val + 1/6*F4val
