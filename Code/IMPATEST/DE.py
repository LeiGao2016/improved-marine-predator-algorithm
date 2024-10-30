import numpy as np
import matplotlib.pyplot as plt
"""
基本差分进化算法 de
"""

MAX_GENERATION = 500  # 最大迭代次数
GENERATION = 0  # 当前代数
BOUND_X = [-100, 100]  # 上下界
L = 4  # 种子的维度
N = 50 # 种群规模
SIGMA = 0.8  # 缩放因子
CR = 0.2 # 交叉概率
BEST_X = 0
FITS = np.zeros(N)
FITS_U = np.zeros(N)
V = np.zeros((N, L))
U = np.zeros((N, L))
E = 0
T = 50
G = np.zeros(T)


# 目标函数
def Func(x):
    dim = x.shape[0]
    o = np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10 * dim)
    return o


if __name__ == '__main__':
    for k in range(T):
        GENERATION = 0  # 当前代数
        BEST_X = 0
        FITS = np.zeros(N)
        FITS_U = np.zeros(N)
        V = np.zeros((N, L))
        U = np.zeros((N, L))
        # 1、初始化
        X = (BOUND_X[1] - BOUND_X[0]) * np.random.random((N, L)) + BOUND_X[0]
        # 找出随机生成的最好的种子
        for i in range(N):
            FITS[i] = Func(X[i])
        # 找出最好的种子 适应度值最小
        BEST_X = X[np.argmin(FITS)]
        list = []

        for i in range(MAX_GENERATION):
            if Func(BEST_X) < E:
                print(GENERATION)
                break
            # 2、变异
            for i in range(N):
                r = np.random.randint(1, N, 2)
                # 变异向量V[i, :]
                # 变异向量的基使用每次适应度值最好的个体BEST_X
                # V[i, :] = BEST_X + SIGMA * (X[r[0], :] - X[r[1], :]) + SIGMA * (X[r[2], :] - X[r[3], :])
                # V[i, :] = BEST_X + SIGMA * ((X[r[0], :] - X[r[1], :]) + (X[r[2], :] - X[r[3], :]))
                # SIGMA过大 无法收敛
                # SIGMA过小 早熟
                V[i, :] = BEST_X + SIGMA * (X[r[0], :] - X[r[1], :])
            # print(V)
            # 3、交叉操作
            for i in range(N):
                jRand = np.floor(np.random.random() * L)
                for j in range(L):
                    # 至少有一个分量对U[i,j]作出了贡献
                    # CR调小 有利于往BEST_X的方向进化
                    if np.random.random() > CR or j == jRand:
                        U[i, j] = X[i, j]
                    else:
                        U[i, j] = V[i, j]
            # 4、计算适应度值+选择操作
            for i in range(N):
                FITS[i] = Func(X[i])
                FITS_U[i] = Func(U[i])
                # 变异的种子表现更好
                if FITS[i] >= FITS_U[i]:
                    X[i, :] = U[i, :]
                    FITS[i] = FITS_U[i]
                    # 是否比BEST_X适应度值更小
                    if FITS_U[i] < Func(BEST_X):
                        BEST_X = U[i, :]
            list.append(Func(BEST_X))
            print("最优值：", Func(BEST_X))
    #     if GENERATION == MAX_GENERATION:
    #         print("当前次未能收敛")
    #     G[k] = GENERATION
    # print("代数：", G)
    # print("平均代数：", np.mean(G))
    LIST = np.array(list)
    x_Max_iteration = np.arange(0,MAX_GENERATION, 1)
    x_Max_iteration = x_Max_iteration.reshape(1, MAX_GENERATION)
    plt.plot(x_Max_iteration[0], LIST, linestyle="-.", color='green', lw=2)
    plt.legend(['GA'], loc='lower left')
    plt.title('F1 Convergence_curve')
    plt.xlabel('Iter')
    plt.ylabel('Convergence value')
    plt.yscale('log')
    plt.show()

import pandas as pd
df9 = pd.DataFrame(LIST)
df9.to_csv('DEdata1.csv',index= False, header= False)
# 不要头文件，不要列索引
df9 = pd.read_csv('DEdata1.csv',header=None)
# 查看保存的文件，不要头文件
