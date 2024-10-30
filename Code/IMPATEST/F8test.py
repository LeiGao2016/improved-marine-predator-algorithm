# 画二维坐标图
# 读取csv并作图
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DEdata = pd.read_csv("F8DEdata.csv", header=None)  # 读取csv数据
GAdata = pd.read_csv("F8GAdata.csv", header=None)
PSOdata = pd.read_csv("F8PSOdata.csv", header=None)
MPAdata = pd.read_csv("F8MPAdata.csv", header=None)
IMPAdata = pd.read_csv("F8HMPAdata.csv", header=None)
plt.plot(DEdata,color = 'blue',marker='o',markevery=25,linestyle="-.")  # 画散点图，*:r表示点用*表示，颜色为红色
plt.plot(GAdata,color = 'orange',marker='v',markevery=25)
plt.plot(PSOdata,color = 'green',marker='s',markevery=25,linestyle="-.")
plt.plot(MPAdata,color = 'magenta',marker='p',markevery=25,linestyle="-.")
plt.plot(IMPAdata,color = 'red',marker='*',markevery=25,linestyle="-.")
plt.legend(['DE','GA','PSO','MPA','IMPA'],loc='lower left')
plt.title("F1:Comparison of optimization results",fontdict = {'family' : 'Times New Roman'})  # 设置标题
plt.xlabel("Iter")  # 横轴名称
plt.ylabel("Fitness value")  # 纵轴名称
# plt.yscale('log')
plt.show()  # 画图