import numpy as np
import levy
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
SearchAgents_no = 800
dim = 1
RL = 0.05*levy.levy(SearchAgents_no, dim, 1.5)
RB = np.random.randn(SearchAgents_no, dim)
# plt.plot(RB)
plt.plot(RL)