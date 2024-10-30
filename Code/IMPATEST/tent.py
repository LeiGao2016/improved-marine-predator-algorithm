import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import initialization as ini
lb = 0
ub = 1
N = 100
D = 4
Alpa=0.499
Z = np.random.rand(N, D)
for i in range(N):
    for j in range(D):
        if Z[i,j]<Alpa:
            Z[i,j] = Z[i,j]/Alpa
        elif Z[i,j]>=Alpa:
            Z[i, j] = (1- Z[i,j])/(1-Alpa)

Z=Z*(ub-lb)+lb
Prey2 = ini.initialization(N, D, ub, lb)

fig = plt.figure(figsize=(8, 4)) # 创建一个图
plt.rcParams['font.family']=['SimHei']#关键是这句left=0.05, right=0.95, bottom=0.05, top=0.9
plt.subplots_adjust(left=0, bottom=0.15, right=None, top=None, wspace=0.03, hspace=0.15)

ax1 = fig.add_subplot(121, projection='3d')
cm = plt.cm.get_cmap('jet')  # 颜色映射，为jet型映射规则
fig1 = ax1.scatter(Z[:,0],Z[:,1],Z[:,2], c =Z[:,3], cmap=cm)
# cb = plt.colorbar(fig1)  # 设置坐标轴
ax1.set_xlabel('x',fontdict = {'family' : 'Times New Roman'})
ax1.set_ylabel('y',fontdict = {'family' : 'Times New Roman'})
ax1.set_zlabel('z',fontdict = {'family' : 'Times New Roman'})
ax1.set_title('(a)    Tent',fontdict = {'family' : 'Times New Roman'})

l = 0.92
b = 0.091
w = 0.009
h = 0.8
#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h]
cbar_ax = fig.add_axes(rect)
cb=plt.colorbar(fig1, cax=cbar_ax)
cb.set_label(label='Particle value',fontdict={'family' : 'Times New Roman'}, loc='top')

ax2 = fig.add_subplot(122, projection='3d')
ax2.set_xlabel('x',fontdict = {'family' : 'Times New Roman'})
ax2.set_ylabel('y',fontdict = {'family' : 'Times New Roman'})
ax2.set_zlabel('z',fontdict = {'family' : 'Times New Roman'})
ax2.set_title('(b)    random',fontdict = {'family' : 'Times New Roman'})
cm = plt.cm.get_cmap('jet')  # 颜色映射，为jet型映射规则
fig2 = ax2.scatter(Prey2[:,0],Prey2[:,1],Prey2[:,2], c =Prey2[:,3], cmap=cm)
# cb = plt.colorbar(fig2)  #设置坐标轴
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.5)
# l = 0.54
# b = 0.091
# w = 0.009
# h = 0.8
# #对应 l,b,w,h；设置colorbar位置；
# rect = [l,b,w,h]
# cbar_ax = fig.add_axes(rect)
# plt.colorbar(fig2, cax=cbar_ax)
