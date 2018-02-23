#!/usr/bin/env python3

import numpy as np
import p2

A = np.matrix([[6,5,0],[5,1,4],[0,4,3]])
Q,R = p2.qrDecomp(A)
print(Q)
print(R)
