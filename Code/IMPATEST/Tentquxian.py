import numpy as np
import matplotlib.pyplot as plt
N = 1000
Alpa=0.499
# D = 4
Z = np.random.rand(N)
for i in range(N):
        if Z[i]<Alpa:
            Z[i] = Z[i]/Alpa
        elif Z[i]>=Alpa:
            Z[i] = (1- Z[i])/(1-Alpa)
plt.subplots_adjust(left=0.125,right=0.9,bottom=0.1,top=0.9,wspace=1,hspace=0.6)
# plt.rcParams['font.family']=['SimHei']
ax1 = plt.subplot(2,1,1)
plt.plot(Z,'.')
plt.title('(a) Tent particle distribution',fontdict = {'family' : 'Times New Roman'})
plt.xlabel('D',fontdict = {'family' : 'Times New Roman'})
plt.ylabel('Tent value',fontdict = {'family' : 'Times New Roman'})

ax2 = plt.subplot(2,1,2)
plt.hist(Z)
plt.title('(b) Tent frequency distribution',fontdict = {'family' : 'Times New Roman'})
plt.xlabel('Tent value',fontdict = {'family' : 'Times New Roman'})
plt.ylabel('Frequency',fontdict = {'family' : 'Times New Roman'})
plt.show()