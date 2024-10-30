# import Get_Functions_details as GFd
# import initialization as ini
# import PSO as pso
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import numpy as np
# import initialization as ini
# import levy
# import random
# import math
# 调出numpy
import numpy as np

df = [-7374.833744569364,-7971.782704909713, -12268.651964468132,-8997.230581628515,-11045.361920967582]  # 要计算的数值

# 求均值
mean = np.mean(df)
# 求方差
var = np.var(df)
# 求标准差
std = np.std(df, ddof=1)

# 数值输出,2f为保留两位小数
print("平均值为：",mean)
print("方 差 为：", var)
print("标准差为：", std)
#
# def fobj(x):
#     o = np.sum(x**2)
#     return o
#
#
# SearchAgents_no = 25
# dim = 50
# Max_iter = 500
# lb = -100
# ub = 100
# c1 = 1.49445
# c2 = 1.49445
# w = 0.729
# v_max = 0.1*ub  # 每个维度粒子的最大速度
# v_min = 0.1*lb
# x = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb  # 初始化每个粒子的位置
# v = (v_max - v_min) * np.random.rand(SearchAgents_no, dim) + v_min
# vx = np.zeros_like(v)
#
# # 初始化每个粒子的适应度值
# g_best = math.inf
# p = x  # 用来存储每个粒子的最佳位置
# p_best = np.ones(SearchAgents_no)  # 用来存储每个粒子的适应度值
# x_best = np.ones(dim)
# for i in range(SearchAgents_no):
#     Flag4ub = (x[i, :] > ub).astype(int)
#     Flag4lb = (x[i, :] < lb).astype(int)  # 就是相当于判断呗
#     x[i, :] = (x[i, :] * (np.logical_not(Flag4ub + Flag4lb).astype(int)) + ub * Flag4ub + lb * Flag4lb)
#     p_best[i] = fobj(x[i, :])
#     if p_best[i] < g_best:
#         g_best = p_best[i]
#         x_best = x[i, :].copy()
#
# gb = np.ones(Max_iter)  # 用来存储每依次迭代的最优值
# for i in range(Max_iter):
#     for j in range(SearchAgents_no):
#         # 更新每个个体最优值和最优位置
#         if p_best[j] > fobj(x[j, :]):
#             p_best[j] = fobj(x[j, :])
#             p[j, :] = x[j, :].copy()
#             # 更新全局最优位置和最优值
#         if p_best[j] < g_best:
#             g_best = p_best[j]
#             x_best = x[j, :].copy()
#         # 更新速度, 因为位置需要后面进行概率判断更新
#         v[j, :] = w * v[j, :] + c1 * np.random.rand(1) * (p_best[j] - x[j, :]) + c2 * np.random.rand(1) * (x_best - x[j, :])
#         x[j, :] = x[j, :] + v[j, :]
#         # 边界条件处理
#         for jj in range(dim):
#             if (v[j, jj] > v_max) or (v[j, jj] < v_min):
#                 v[j, jj] = v_min + np.random.rand(1) * (v_max - v_min)
#         vx[j, :] = 1 / (1 + np.exp(-v[j, :]))
#         for ii in range(dim):
#             x[j, ii] = 1 if vx[j, ii] > np.random.rand(1) else 0
#     gb[i] = g_best
#
# # Top_predator_pos = np.zeros(([1, dim]))
# # Top_predator_fit = math.inf
# # Convergence_curve = np.zeros(([1, Max_iter]))
# # stepsize = np.zeros((SearchAgents_no, dim))
# # fitness = np.full((SearchAgents_no, 1), np.inf)
# #
# # Prey = ini.initialization(SearchAgents_no, dim, ub, lb)
# # Xmin = np.ones((SearchAgents_no, dim))*lb
# # Xmax = np.ones((SearchAgents_no, dim))*ub
# # Iter = 0
# # FADs = 0.2
# # P = 0.5
# # while Iter < Max_iter:
# #         # print('Iter:', Iter)
# #     '''# %------------------- Detecting top predator1 -----------------'''
# #     for i in range(Prey.shape[0]):
# #             #边界范围的设置
# #         Flag4ub = (Prey[i, :] > ub).astype(int)
# #         Flag4lb = (Prey[i, :] < lb).astype(int)#就是相当于判断呗
# #         Prey[i, :] = (Prey[i, :]*(np.logical_not(Flag4ub+Flag4lb).astype(int))+ub*Flag4ub+lb*Flag4lb)
# #         fitness[i, 0] = fobj(Prey[i, :].reshape(1, Prey[i, :].shape[0]))
# #             # print((Prey[i, :].reshape(1, Prey[i, :].shape[0])).shape)
# #         if(fitness[i, 0] < Top_predator_fit):
# #             Top_predator_fit = fitness[i, 0]
# #             Top_predator_pos = Prey[i].reshape((Top_predator_pos.shape))
# #         '''   # %------------------- Marine Memory saving1 -------------------'''
# #
# #     if Iter == 0:
# #         fit_old = fitness
# #         Prey_old = Prey
# #             # print('fit_old_0', fit_old)
# #     Inx = np.zeros(fitness.shape[0]).reshape(fitness.shape[0], 1)
# #     for i in range(fitness.shape[0]):
# #         if(fit_old[i] < fitness[i]):
# #             Inx[i] = 0
# #         else:
# #             Inx[i] = 1
# #         # print(Inx)
# #     Indx = np.full((Inx.shape[0], dim), Inx).astype(int)
# #     Prey = Indx*Prey_old + np.logical_not(Indx).astype(int) * Prey
# #     fitness = Inx*fit_old + np.logical_not(Inx).astype(int) * fitness
# #     fit_old = fitness
# #     Prey_old = Prey
# #     '''#  %------------------------------------------------------------'''
# #     Elite = np.full((SearchAgents_no, Top_predator_pos.shape[1]), Top_predator_pos)  # %(Eq. 10)
# #     CF = (1-Iter/Max_iter)**(2*Iter/Max_iter)
# #     RL = 0.05*levy.levy(SearchAgents_no, dim, 1.5)
# #     RB = np.random.randn(SearchAgents_no, dim)
# #     for i in range(Prey.shape[0]):
# #         for j in range(Prey.shape[1]):
# #             R = random.uniform(0, 1)
# #             # print(R)
# #             #  %------------------ Phase 1 (Eq.12) -------------------
# #             if Iter < Max_iter/3:
# #                 stepsize[i, j] = RB[i, j]*(Elite[i, j]-RB[i, j]*Prey[i, j])
# #                 Prey[i, j] = Prey[i, j]+P*R*stepsize[i, j]
# #             # %--------------- Phase 2 (Eqs. 13 & 14)----------------
# #             elif (Iter > Max_iter/3) and (Iter < 2*Max_iter/3):
# #                 if i > Prey.shape[0]/2:
# #                     stepsize[i, j] = RB[i, j] * \
# #                         (RB[i, j]*Elite[i, j]-Prey[i, j])
# #                     Prey[i, j] = Elite[i, j]+P*CF*stepsize[i, j]
# #                 else:
# #                     stepsize[i, j] = RL[i, j] * \
# #                                      (Elite[i, j]-RL[i, j]*Prey[i, j])
# #                     Prey[i, j] = Prey[i, j]+P*R*stepsize[i, j]
# #                 #  %----------------- Phase 3 (Eq. 15)-------------------
# #             else:
# #                 stepsize[i, j] = RL[i, j]*(RL[i, j]*Elite[i, j]-Prey[i, j])
# #                 Prey[i, j] = Elite[i, j]+P*CF*stepsize[i, j]
# #         '''# %------------------- Detecting top predator2 -----------------'''
# #     for i in range(Prey.shape[0]):
# #         Flag4ub = (Prey[i, :] > ub).astype(int)
# #         Flag4lb = (Prey[i, :] < lb).astype(int)
# #         Prey[i, :] = (
# #             Prey[i, :]*(np.logical_not(Flag4ub+Flag4lb).astype(int))+ub*Flag4ub+lb*Flag4lb)
# #
# #         fitness[i, 0] = fobj(Prey[i, :])
# #
# #         if(fitness[i, 0] < Top_predator_fit):
# #             Top_predator_fit = fitness[i, 0]
# #             Top_predator_pos = Prey[i].reshape((Top_predator_pos.shape))
# #         '''# %---------------------- Marine Memory saving2 ----------------'''
# #     if Iter == 0:
# #         fit_old = fitness
# #         Prey_old = Prey
# #             # print('fit_old_0', fit_old)
# #     Inx = np.zeros(fitness.shape[0]).reshape(fitness.shape[0], 1)
# #     for i in range(fitness.shape[0]):
# #         if(fit_old[i] < fitness[i]):
# #             Inx[i] = 0
# #         else:
# #             Inx[i] = 1
# #         # print(Inx)
# #     Indx = np.full((Inx.shape[0], dim), Inx).astype(int)
# #     Prey = Indx*Prey_old + np.logical_not(Indx).astype(int) * Prey
# #     fitness = Inx*fit_old + np.logical_not(Inx).astype(int) * fitness
# #     fit_old = fitness
# #     Prey_old = Prey
# #
# #     '''#  %---------- Eddy formation and FADs� effect (Eq 16) -----------'''
# #     if random.uniform(0, 1) < FADs:
# #         U = np.random.rand(SearchAgents_no, dim) < FADs
# #         Prey = Prey+CF * \
# #                ((Xmin+np.random.rand(SearchAgents_no, dim)*(Xmax-Xmin))*U)
# #     else:
# #         r = random.uniform(0, 1)
# #         Rs = Prey.shape[0]
# #         stepsize = (
# #             FADs*(1-r)+r)*(Prey[np.random.permutation(Rs), :]-Prey[np.random.permutation(Rs), :])
# #         Prey = Prey+stepsize
# #     Iter = Iter+1
# #     Convergence_curve[:, Iter-1] = Top_predator_fit
# #     # print('fitness', fitness)
