import numpy as np
import matplotlib.pyplot as plt
Max_iter = 500
P = np.zeros( Max_iter)
for Iter in range(Max_iter):
    P[Iter] = np.exp(np.tan(-np.pi/6*np.exp(Iter/Max_iter)))
print(P)

plt.plot(P)
plt.title(' The change of step size P of predator',fontdict = {'family' : 'Times New Roman'})
plt.ylabel('P value')
plt.xlabel('Iter')
