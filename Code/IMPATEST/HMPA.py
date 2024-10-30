import numpy as np
# import initialization as ini
from IMPAAAAA.MPATEST import initialization as ini
# import levy
from IMPAAAAA.MPATEST import levy
import random
import math
import pandas as pd

def MPA(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    Top_predator_pos = np.zeros(([1, dim]))
    Top_predator_fit = math.inf
    Convergence_curve = np.zeros(([1, Max_iter]))
    stepsize = np.zeros((SearchAgents_no, dim))
    fitness = np.full((SearchAgents_no, 1), np.inf)

    Z = np.random.rand(SearchAgents_no,dim)
    for i in range(SearchAgents_no):
        for j in range(dim):
            if Z[i,j]<0.5:
                Z[i,j] = 2*Z[i,j]
            elif Z[i,j]>=0.5:
                Z[i, j] = 2 *(1- Z[i, j])

    Prey = lb+Z*(ub-lb)
    Xmin = np.ones((SearchAgents_no, dim)) * lb
    Xmax = np.ones((SearchAgents_no, dim)) * ub
    Iter = 0
    FADs = 0.2

    while Iter < Max_iter:
        # print('Iter:', Iter)
        '''# %------------------- Detecting top predator1 -----------------'''
        for i in range(Prey.shape[0]):
            #设置种群范围，将超过边境的粒子作处理
            #将矩阵Prey里面的每个元素分别与ub进行比较，如果大于ub那么为1，小于ub则为0,构成一个二进制的Flag4ub的数组
            Flag4ub = (Prey[i, :] > ub).astype(int)
            Flag4lb = (Prey[i, :] < lb).astype(int)

            Prey[i, :] = (Prey[i, :]*(np.logical_not(Flag4ub+Flag4lb).astype(int))+ub*Flag4ub+lb*Flag4lb)
            fitness[i, 0] = fobj(Prey[i, :])
            # print((Prey[i, :].reshape(1, Prey[i, :].shape[0])).shape)
            if(fitness[i, 0] < Top_predator_fit):
                Top_predator_fit = fitness[i, 0]
                Top_predator_pos = Prey[i].reshape((Top_predator_pos.shape))
        '''   # %------------------- Marine Memory saving1 -------------------'''

        if Iter == 0:
            fit_old = fitness
            Prey_old = Prey
            # print('fit_old_0', fit_old)
        Inx = np.zeros(fitness.shape[0]).reshape(fitness.shape[0], 1)
        for i in range(fitness.shape[0]):
            if(fit_old[i] < fitness[i]):
                Inx[i] = 0
            else:
                Inx[i] = 1
        # print(Inx)
        Indx = np.full((Inx.shape[0], dim), Inx).astype(int)
        Prey = Indx*Prey_old + np.logical_not(Indx).astype(int) * Prey
        fitness = Inx*fit_old + np.logical_not(Inx).astype(int) * fitness
        fit_old = fitness
        Prey_old = Prey
        '''#  %------------------------------------------------------------'''
        Elite = np.full(
            (SearchAgents_no, Top_predator_pos.shape[1]), Top_predator_pos)  # %(Eq. 10)
        CF = (1-Iter/Max_iter)**(2*Iter/Max_iter)
        RL = 0.05*levy.levy(SearchAgents_no, dim, 1.5)
        RB = np.random.randn(SearchAgents_no, dim)
        # P = 0.5
        P = np.exp(np.tan(-np.pi/6*np.exp(Iter/Max_iter)))
        c1 = 2.5
        c2 = 0.5
        C = (c1 - c2) * (Max_iter - Iter) / Max_iter + c2
        s = 1 - Iter / Max_iter
        m = np.random.random()
        c_m = np.tanh(m-1/2)*np.pi
        for i in range(Prey.shape[0]):
            for j in range(Prey.shape[1]):
                R = random.uniform(0, 1)
                # print(R)
                #  %------------------ Phase 1 (Eq.12) -------------------
                if Iter < Max_iter/3:
                    stepsize[i, j] = RB[i, j]*(Elite[i, j]-RB[i, j]*Prey[i, j])
                    Prey[i, j] = Prey[i, j]+P*R*stepsize[i, j]
                    Prey[i, j] = Prey[i, j] + Prey[i, j]*c_m
                # %--------------- Phase 2 (Eqs. 13 & 14)----------------
                elif (Iter > Max_iter / 3) and (Iter < 2 * Max_iter / 3):
                    if i > Prey.shape[0] / 2:
                        stepsize[i, j] = s * Prey[i, j] + C * random.random() * (Elite[i, j] - Prey[i, j])
                        Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]

                    else:
                        stepsize[i, j] = RL[i, j] * (Elite[i, j] - RL[i, j] * Prey[i, j])
                        Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]
                        Prey[i, j] = Prey[i, j] + Prey[i, j] * c_m
                #  %----------------- Phase 3 (Eq. 15)-------------------
                else:
                    stepsize[i, j] = s * Prey[i, j] + C * random.random() * (Elite[i, j] - Prey[i, j])
                    Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]
                    # stepsize[i, j] = RL[i, j] * (Elite[i, j] - RL[i, j] * Prey[i, j])
                    # Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]
        '''# %------------------- Detecting top predator2 -----------------'''
        for i in range(Prey.shape[0]):
            Flag4ub = (Prey[i, :] > ub).astype(int)
            Flag4lb = (Prey[i, :] < lb).astype(int)
            Prey[i, :] = (Prey[i, :]*(np.logical_not(Flag4ub+Flag4lb).astype(int))+ub*Flag4ub+lb*Flag4lb)
            fitness[i, 0] = fobj(Prey[i, :])

            if (fitness[i, 0] < Top_predator_fit):
                Top_predator_fit = fitness[i, 0]
                Top_predator_pos = Prey[i].reshape((Top_predator_pos.shape))
        '''# %---------------------- Marine Memory saving2 ----------------'''
        if Iter == 0:
            fit_old = fitness
            Prey_old = Prey
            # print('fit_old_0', fit_old)
        Inx = np.zeros(fitness.shape[0]).reshape(fitness.shape[0], 1)
        for i in range(fitness.shape[0]):
            if(fit_old[i] < fitness[i]):
                Inx[i] = 0
            else:
                Inx[i] = 1
        # print(Inx)
        Indx = np.full((Inx.shape[0], dim), Inx).astype(int)
        Prey = Indx*Prey_old + np.logical_not(Indx).astype(int) * Prey
        fitness = Inx*fit_old + np.logical_not(Inx).astype(int) * fitness
        fit_old = fitness
        Prey_old = Prey

        '''#  %---------- Eddy formation and FADs� effect (Eq 16) -----------'''
        if random.uniform(0, 1) < FADs:
            U = np.random.rand(SearchAgents_no, dim) < FADs
            Prey = Prey+CF * \
                ((Xmin+np.random.rand(SearchAgents_no, dim)*(Xmax-Xmin))*U)
        else:
            r = random.uniform(0, 1)
            Rs = Prey.shape[0]
            stepsize = (
                FADs*(1-r)+r)*(Prey[np.random.permutation(Rs), :]-Prey[np.random.permutation(Rs), :])
            Prey = Prey+stepsize
        Iter = Iter+1
        Convergence_curve[:, Iter-1] = Top_predator_fit
        # print('fitness', fitness)

    return {'Top_predator_fit': Top_predator_fit, 'Top_predator_pos': Top_predator_pos, 'Convergence_curve': Convergence_curve, }