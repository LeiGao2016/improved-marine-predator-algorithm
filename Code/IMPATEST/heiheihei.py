import numpy as np
import initialization as ini
import levy
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
Max_iter = 600
m = np.random.random()
for Iter in range(Max_iter):
    c_m = np.tanh(m-1/2)*np.pi
x_Max_iteration = np.arange(0, Max_iter, 1)
plt.plot(x_Max_iteration[0], c_m, linestyle="-.",color='blue', lw=2)