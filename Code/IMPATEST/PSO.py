import numpy as np
import math
import matplotlib.pyplot as plt


def PSO(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    c1 = 1
    c2 = 0.5
    w = 0.3
    # w_max = 0.8
    # w_min = 0.4
    v_max = 1  # 每个维度粒子的最大速度
    v_min = -1
    x = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb  # 初始化每个粒子的位置
    v = np.random.rand(SearchAgents_no, dim) * (v_max - v_min) + v_min # 初始化每个粒子的速度

    # 初始化每个粒子的适应度值
    g_best = 100
    gb = np.ones(Max_iter)  # 用来存储每依次迭代的最优值
    p = x  # 用来存储每个粒子的最佳位置
    p_best = np.ones((SearchAgents_no,1))  # 用来存储每个粒子的适应度值
    x_best = np.ones(dim)
    for i in range(SearchAgents_no):
        p_best[i] = fobj(x[i, :])
        # p[i,:] = x[i,:]

    for i in range(Max_iter):
        for j in range(SearchAgents_no):
            # 更新每个个体最优值和最优位置
            if p_best[j] > fobj(x[j, :]):
                p_best[j] = fobj(x[j, :])
                p[j, :] = x[j, :].copy()
            # 更新全局最优位置和最优值
            if p_best[j] < g_best:
                g_best = p_best[j]
                x_best = x[j, :].copy()
            # w = w_max - (w_max - w_min) * i / Max_iter
            # 更新速度, 因为位置需要后面进行概率判断更新
            v[j, :] = w * v[j, :] + c1 * np.random.rand(1) * (p[j,:] - x[j, :]) + c2 * np.random.rand(1) * (x_best - x[j, :])
            x[j, :] = x[j, :] + v[j, :]
            # 边界条件处理
            for jj in range(dim):
                if (v[j, jj] > v_max) or (v[j, jj] < v_min):
                    v[j, jj] = v_min + np.random.rand(1) * (v_max - v_min)
                if (x[j, jj] > ub) or (x[j, jj] < lb):
                    x[j, jj] = lb + np.random.rand(1) * (ub - lb)

            # vx[j, :] = 1 / (1 + np.exp(-v[j, :]))
            # for ii in range(dim):
            #     x[j, ii] = 1 if vx[j, ii] > np.random.rand(1) else 0
        gb[i] = g_best

    return {'PSO_fit': gb[Max_iter-1], 'PSO_pos': x_best, 'Convergence_curve': gb, }



import Get_Functions_details as GFd
import pandas as pd
SearchAgents_no = 10  # Number of search agents
Function_name = "F10"
Max_iteration = 500
a = ''.join([i for i in Function_name if i.isdigit()])
ludf = GFd.switcher(int(a))
lb, ub, dim, fobj = ludf['lb'], ludf['ub'], ludf['dim'], ludf['fobj']
kq1 = PSO(SearchAgents_no, Max_iteration, lb, ub, dim, fobj)
Best_score1, Best_pos1, Convergence_curve1 = kq1['PSO_fit'], kq1[
    'PSO_pos'], kq1['Convergence_curve']
x_Max_iteration = np.arange(0, Max_iteration, 1)
x_Max_iteration = x_Max_iteration.reshape(1, Max_iteration)
# plt.plot(x_Max_iteration[0], Convergence_curve[0], color='blue', lw=2)
plt.plot(x_Max_iteration[0], Convergence_curve1, linestyle="-.",color='blue', lw=2)
plt.yscale('log')
df4 = pd.DataFrame(Convergence_curve1)
df4.to_csv('GAdata1.csv',index= False, header= False)
# 不要头文件，不要列索引
df5 = pd.read_csv('GAdata1.csv',header=None)