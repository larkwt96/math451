#!/usr/bin/env python3

from p1 import *

if __name__ == '__main__':
    d01 = 200.1
    d02 = 210
    exp_val = np.float64(325306921.75922555)
    p1_count = 0
    p2_count = 0
    print('Expected value: {}'.format(exp_val))
    for i in range(1, 30):
        (dt1, tk1, xk1) = run(dt=200/i)
        (dt2, tk2, xk2) = run(dt=210/i)
        xk1 = xk1[0]
        xk2 = xk2[0]
        err1 = xk1-exp_val
        err2 = xk2-exp_val
        print("dt: {} d1(36000): {} err: {}".format(dt1, xk1, err1))
        print("dt: {} d2(36000): {} err: {}".format(dt2, xk2, err2))
        print("d1 more accurate than d2: {}".format(abs(err1) < abs(err2)))
        if (abs(err1) < abs(err2)):
            p1_count += 1
        else:
            p2_count += 1
    print("Where delta t divides 600, it was more accurate {} times.".format(p1_count))
    print("Where delta t doesn't divide 600, it was more accurate {} times.".format(p2_count))
