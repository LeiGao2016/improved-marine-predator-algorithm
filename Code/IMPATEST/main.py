import Get_Functions_details as GFd
import PSO as pso
import MPA as mpa
import HMPA as hmpa
import numpy as np
import matplotlib.pyplot as plt
import time
t1 = time.time()
# fobj = @YourCostFunction
# dim = number of your variables
# Max_iteration = maximum number of iterations
# SearchAgents_no = number of search agents种群数量
# lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
# ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
SearchAgents_no = 80  # Number of search agents
Function_name = "F12"
Max_iteration = 500
a = ''.join([i for i in Function_name if i.isdigit()])
ludf = GFd.switcher(int(a))
lb, ub, dim, fobj = ludf['lb'], ludf['ub'], ludf['dim'], ludf['fobj']
print(lb, ub, dim, fobj)
kq1 = pso.PSO(SearchAgents_no, Max_iteration, lb, ub, dim, fobj)
Best_score1, Best_pos1, Convergence_curve1 = kq1['PSO_fit'], kq1[
    'PSO_pos'], kq1['Convergence_curve']
kq2 = mpa.MPA(SearchAgents_no, Max_iteration, lb, ub, dim, fobj)
Best_score2, Best_pos2, Convergence_curve2 = kq2['Top_predator_fit'], kq2[
    'Top_predator_pos'], kq2['Convergence_curve']
kq3 = hmpa.MPA(SearchAgents_no, Max_iteration, lb, ub, dim, fobj)
Best_score3, Best_pos3, Convergence_curve3 = kq3['Top_predator_fit'], kq3[
    'Top_predator_pos'], kq3['Convergence_curve']
# print("Best_score", Best_score)
# print("Best_pos", Best_pos)
# print("Convergence_curve", Convergence_curve)
x_Max_iteration = np.arange(0, Max_iteration, 1)
x_Max_iteration = x_Max_iteration.reshape(1, Max_iteration)
# plt.plot(x_Max_iteration[0], Convergence_curve[0], color='blue', lw=2)
plt.plot(x_Max_iteration[0], Convergence_curve1, linestyle="-.",color='blue', lw=2)
plt.plot(x_Max_iteration[0], Convergence_curve2[0], linestyle="-.",color='purple', lw=2)
plt.plot(x_Max_iteration[0], Convergence_curve3[0], color='red', lw=2)
plt.legend(['PSO','MPA','HMPA'],loc='lower left')
plt.title('F1 Convergence_curve')
plt.xlabel('Iter')
plt.ylabel('Convergence value')
# plt.yscale('log')
plt.show()
# print("Best_score", Best_score)
# print("kt")
# t2 = time.time()
# print('t:',t2-t1)
import pandas as pd
df1 = pd.DataFrame(Convergence_curve1)
df1.to_csv('F8DEdata.csv',index= False, header= False)
df1 = pd.read_csv('F8DEdata.csv',header=None)
# df1 = pd.DataFrame(Convergence_curve2[0])
# df1.to_csv('F8MPAdata.csv',index= False, header= False)
# df1 = pd.read_csv('F8MPAdata.csv',header=None)
# df1 = pd.DataFrame(Convergence_curve3[0])
# df1.to_csv('F8HMPAdata.csv',index= False, header= False)
# df1 = pd.read_csv('F8HMPAdata.csv',header=None)

