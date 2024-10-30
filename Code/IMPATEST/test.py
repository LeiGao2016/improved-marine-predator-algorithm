import numpy as np
import matplotlib.pyplot as plt
import random
N = 25  # Number of search agents
Max_iteration = 500
D=30
Z = np.random.rand(N,D)
for i in range(N):
    for j in range(D):
        if Z[i,j]<0.5:
            Z[i,j] = 2*Z[i,j]
        elif Z[i,j]>=0.5:
            Z[i, j] = 2 *(1- Z[i, j])
ub = 100
lb = -100
Prey = lb+Z*(ub-lb)


