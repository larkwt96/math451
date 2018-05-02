#!/usr/bin/env python3
def step(f, dt, tk1, xk1):
    arg1 = tk1+.5*dt
    arg2 = xk1 + .5*dt*f(tk1, xk1)
    tk = tk1 + dt
    xk = xk1 + dt*f(arg1, arg2)
    return tk, xk
