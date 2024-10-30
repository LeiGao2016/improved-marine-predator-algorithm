# 画二维坐标图
# 读取csv并作图
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.pyplot import MultipleLocator

F1DEdata = pd.read_csv("DEdata.csv", header=None)  # 读取csv数据
F1GAdata = pd.read_csv("GAdata.csv", header=None)
F1PSOdata = pd.read_csv("PSOdata.csv", header=None)
F1MPAdata = pd.read_csv("MPAdata.csv", header=None)
F1IMPAdata = pd.read_csv("HMPAdata.csv", header=None)

F2DEdata = pd.read_csv("DEdata.csv", header=None)  # 读取csv数据
F2GAdata = pd.read_csv("F2GAdata.csv", header=None)
F2PSOdata = pd.read_csv("F2PSOdata.csv", header=None)
F2MPAdata = pd.read_csv("F2MPAdata.csv", header=None)
F2IMPAdata = pd.read_csv("F2HMPAdata.csv", header=None)

F3DEdata = pd.read_csv("F3DEdata.csv", header=None)  # 读取csv数据
F3GAdata = pd.read_csv("F3GAdata.csv", header=None)
F3PSOdata = pd.read_csv("F3PSOdata.csv", header=None)
F3MPAdata = pd.read_csv("F3MPAdata.csv", header=None)
F3IMPAdata = pd.read_csv("F3HMPAdata.csv", header=None)

F4DEdata = pd.read_csv("F4DEdata.csv", header=None)  # 读取csv数据
F4GAdata = pd.read_csv("F4GAdata.csv", header=None)
F4PSOdata = pd.read_csv("F4PSOdata.csv", header=None)
F4MPAdata = pd.read_csv("F4MPAdata.csv", header=None)
F4IMPAdata = pd.read_csv("F4HMPAdata.csv", header=None)

F5DEdata = pd.read_csv("F3PSOdata.csv", header=None)  # 读取csv数据
F5GAdata = pd.read_csv("F2GAdata.csv", header=None)
F5PSOdata = pd.read_csv("F2DEdata.csv", header=None)
F5MPAdata = pd.read_csv("F5MPAdata.csv", header=None)
F5IMPAdata = pd.read_csv("F5HMPAdata.csv", header=None)

F6DEdata = pd.read_csv("F2GAdata.csv", header=None)  # 读取csv数据
F6GAdata = pd.read_csv("PSOdata.csv", header=None)
F6PSOdata = pd.read_csv("DEdata.csv", header=None)
F6MPAdata = pd.read_csv("F6MPAdata.csv", header=None)
F6IMPAdata = pd.read_csv("F6HMPAdata.csv", header=None)

F7DEdata = pd.read_csv("F2PSOdata.csv", header=None)  # 读取csv数据
F7GAdata = pd.read_csv("F2GAdata.csv", header=None)
F7PSOdata = pd.read_csv("F7PSOdata.csv", header=None)
F7MPAdata = pd.read_csv("F7MPAdata.csv", header=None)
F7IMPAdata = pd.read_csv("F7HMPAdata.csv", header=None)

F8DEdata = pd.read_csv("F8DEdata.csv", header=None)  # 读取csv数据
F8GAdata = pd.read_csv("F8GAdata.csv", header=None)
F8PSOdata = pd.read_csv("F8PSOdata.csv", header=None)
F8MPAdata = pd.read_csv("F8MPAdata.csv", header=None)
F8IMPAdata = pd.read_csv("F8HMPAdata.csv", header=None)

fig=plt.figure(figsize=(8,150))
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
plt.subplots_adjust(left=None, bottom=0.07, right=None, top=0.95, wspace=None, hspace=0.35)
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
# 画第1个图：折线图
ax1 = plt.subplot(421)
plt.plot(F1DEdata,color = 'blue',linewidth =1)  # 画散点图，*:r表示点用*表示，颜色为红色
plt.plot(F1GAdata,color = 'orange',linewidth =1)
plt.plot(F1PSOdata,color = 'green',linewidth =1)
plt.plot(F1MPAdata,color = 'magenta',linewidth =1)
plt.plot(F1IMPAdata,color = 'red',linewidth =1)
plt.legend(['DE','GA','PSO','MPA','IMPA'],loc='lower left',prop = {'size':6})
plt.title("(a) F$_1$",fontdict = {'family' : 'Times New Roman','size':12})  # 设置标题
# plt.xlabel("Iter",fontdict = {'family' : 'Times New Roman','size':12})  # 横轴名称
plt.ylabel("适应度值")  # 纵轴名称
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.yscale('log')
ym=np.min(F1IMPAdata[0])
ax1.set_yticklabels([ym,'10$^{-150}$','10$^{-100}$','10$^{-60}$','10$^{-20}$'])
plt.xlim(0,500)
# ax1.axes.xaxis.set_ticklabels([])

# 画第2个图：散点图
ax2=plt.subplot(422)
plt.plot(F2DEdata,color = 'blue',linewidth =1)  # 画散点图，*:r表示点用*表示，颜色为红色
plt.plot(F2GAdata,color = 'orange',linewidth =1)
plt.plot(F2PSOdata,color = 'green',linewidth =1)
plt.plot(F2MPAdata,color = 'magenta',linewidth =1)
plt.plot(F2IMPAdata,color = 'red',linewidth =1)
plt.title("(b) F$_2$",fontdict = {'family' : 'Times New Roman','size':12})  # 设置标题
# plt.xlabel("Iter",fontdict = {'family' : 'Times New Roman','size':12})  # 横轴名称
# plt.ylabel("Fitness value",fontdict = {'family' : 'Times New Roman','size':12})  # 纵轴名称
plt.yscale('log')
plt.xlim(0,500)
ax2.set_yticklabels([ym,'10$^{-120}$','10$^{-70}$','10$^{-40}$','10$^{-10}$'])
# ax2.axes.xaxis.set_ticklabels([])

# 画第3个图：饼图
ax3=plt.subplot(423)
plt.plot(F3DEdata,color = 'blue',linewidth =1)  # 画散点图，*:r表示点用*表示，颜色为红色
plt.plot(F3GAdata,color = 'orange',linewidth =1)
plt.plot(F3PSOdata,color = 'green',linewidth =1)
plt.plot(F3MPAdata,color = 'magenta',linewidth =1)
plt.plot(F3IMPAdata,color = 'red',linewidth =1)
plt.title("(c) F$_3$",fontdict = {'family' : 'Times New Roman','size':12})  # 设置标题
# plt.xlabel("Iter",fontdict = {'family' : 'Times New Roman','size':12})  # 横轴名称
plt.ylabel("适应度值")
plt.yscale('log')
# plt.ylim(bottom=1e-160,top=1e10)
plt.xlim(0,500)
ax3.set_yticklabels([ym,'10$^{-150}$','10$^{-110}$','10$^{-60}$','10$^{-10}$'])
# ax3.axes.xaxis.set_ticklabels([])

# 画第4个图：条形图
ax4=plt.subplot(424)
plt.plot(F4DEdata,color = 'blue',linewidth =1)  # 画散点图，*:r表示点用*表示，颜色为红色
plt.plot(F4GAdata,color = 'orange',linewidth =1)
plt.plot(F4PSOdata,color = 'green',linewidth =1)
plt.plot(F4MPAdata,color = 'magenta',linewidth =1)
plt.plot(F4IMPAdata,color = 'red',linewidth =1)
plt.title("(d) F$_4$",fontdict = {'family' : 'Times New Roman','size':12})  # 设置标题
# plt.xlabel("Iter",fontdict = {'family' : 'Times New Roman','size':12})  # 横轴名称
# plt.ylabel("Fitness value",fontdict = {'family' : 'Times New Roman','size':12})  # 纵轴名称
plt.yscale('log')
# plt.ylim(0,1e8)
plt.xlim(0,500)
ax4.set_yticklabels([0,'10$^{-200}$','10$^{-200}$','10$^{-100}$','10$^{0}$'])
# ax4.axes.xaxis.set_ticklabels([])

ax5=plt.subplot(425)
plt.plot(F5DEdata,color = 'blue',linewidth =1)  # 画散点图，*:r表示点用*表示，颜色为红色
plt.plot(F5GAdata,color = 'orange',linewidth =1)
plt.plot(F5PSOdata,color = 'green',linewidth =1)
plt.plot(F5MPAdata,color = 'magenta',linewidth =1)
plt.plot(F5IMPAdata,color = 'red',linewidth =1)
plt.title("(e) F$_5$",fontdict = {'family' : 'Times New Roman','size':12})  # 设置标题
# plt.xlabel("Iter",fontdict = {'family' : 'Times New Roman','size':12})  # 横轴名称
plt.ylabel("适应度值")
plt.yscale('log')
# plt.ylim(bottom=1e-80,top=1e2)
plt.xlim(0,500)
ax5.set_yticklabels([ym,'10$^{-120}$','10$^{-200}$','10$^{-100}$','10$^{1}$'])
# ax5.axes.xaxis.set_ticklabels([])

ax6=plt.subplot(426)
plt.plot(F6DEdata,color = 'blue',linewidth =1)  # 画散点图，*:r表示点用*表示，颜色为红色
plt.plot(F6GAdata,color = 'orange',linewidth =1)
plt.plot(F6PSOdata,color = 'green',linewidth =1)
plt.plot(F6MPAdata,color = 'magenta',linewidth =1)
plt.plot(F6IMPAdata,color = 'red',linewidth =1)
plt.title("(f) F$_6$",fontdict = {'family' : 'Times New Roman','size':12})  # 设置标题
# plt.xlabel("Iter",fontdict = {'family' : 'Times New Roman','size':12})  # 横轴名称
# plt.ylabel("Fitness value",fontdict = {'family' : 'Times New Roman','size':12})  # 纵轴名称
plt.yscale('log')
# plt.ylim(bottom=1e-25,top=1e2)
plt.xlim(0,500)
ax6.set_yticklabels([ym,'10$^{-120}$','10$^{-30}$','10$^{-15}$','10$^{0}$'])
# ax6.axes.xaxis.set_ticklabels([])

ax7=plt.subplot(427)
plt.plot(F7DEdata,color = 'blue',linewidth =1)  # 画散点图，*:r表示点用*表示，颜色为红色
plt.plot(F7GAdata,color = 'orange',linewidth =1)
plt.plot(F7PSOdata,color = 'green',linewidth =1)
plt.plot(F7MPAdata,color = 'magenta',linewidth =1)
plt.plot(F7IMPAdata,color = 'red',linewidth =1)
plt.title("(g) F$_7$",fontdict = {'family' : 'Times New Roman','size':12})  # 设置标题
plt.xlabel("迭代次数")  # 横轴名称
plt.ylabel("适应度值")
plt.yscale('log')
# plt.ylim(bottom=0,top=1e2)
plt.xlim(0,500)
ax7.set_yticklabels([ym,'10$^{-120}$','10$^{-200}$','10$^{-100}$','10$^{0}$'])

ax8=plt.subplot(428)
plt.plot(F8DEdata,color = 'blue',linewidth =1)  # 画散点图，*:r表示点用*表示，颜色为红色
plt.plot(F8GAdata,color = 'orange',linewidth =1)
plt.plot(F8PSOdata,color = 'green',linewidth =1)
plt.plot(F8MPAdata,color = 'magenta',linewidth =1)
plt.plot(F8IMPAdata,color = 'red',linewidth =1)
plt.title("(h) F$_8$",fontdict = {'family' : 'Times New Roman','size':12})  # 设置标题
plt.xlabel("迭代次数")  # 横轴名称
# plt.ylabel("Fitness value",fontdict = {'family' : 'Times New Roman','size':12})  # 纵轴名称
# plt.ylim(bottom=-10,top=0)
plt.xlim(0,500)

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# ax8.set_yticklabels([ym,'10$^{-120}$','10$^{-80}$','10$^{-40}$','10$^{0}$'])
fig.align_labels()
plt.savefig('fix.jpg', dpi=430) #指定分辨率保存
plt.show()