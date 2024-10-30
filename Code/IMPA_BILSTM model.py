from tensorflow.python.keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply
from tensorflow.python.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import merge
from tensorflow.python.keras.layers.core import *
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.models import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from MPATEST import HMPA as hmpa
import keras.backend as K
from sklearn.metrics import r2_score

input_dim = 3
time_step = 20
batch_size = 36
epochs = 100

df1 = pd.read_csv('D:/pycharm-pytorch/IMPAAAAA/nlm_rawdata(1).csv')
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(df1)
# 下面可以开始神经网络的部分啦！！
train = data[0:4300+time_step]
test = data[4300-time_step:]
print(train.shape)
print(test.shape)

x_train = [] # list
y_train = []
for i in range(time_step, len(train)):  # range(start, stop, step)
    x_train.append(train[i-time_step:i])
    y_train.append(train[i, 0])  # 这个步骤利用append()把二维的变成三维的，以便于输入神经网络。
x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)

#scale the test data
x_test = []
y_test = []
for i in range(time_step, len(test)):
    x_test.append(test[i-time_step:i])
    y_test.append(test[i, 0])
x_test = np.array(x_test)
y_test = np.array(y_test)
print(x_test.shape)
print(y_test.shape)
# 下面可以开始神经网络的部分啦！！

def f1(x):
    neurons1 = int(x[0])
    neurons2 = int(x[1])
    batch_size = int(x[2])
    learn_rate = x[3]
    print(x)
    # x_train, y_train, x_test, y_test = data()
    df1 = pd.read_csv('D:/pycharm-pytorch/IMPAAAAA/nlm_rawdata(1).csv')
    model = Sequential()
    model.add(Bidirectional(
        LSTM(units=neurons1, return_sequences=True, input_dim=x_train.shape[-1], input_length=x_train.shape[1])))
    model.add(Bidirectional(LSTM(units=neurons2)))
    model.add(Dropout(learn_rate))
    model.add(Dense(units=1))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    vis = model.predict(x_test)
    df1 = df1.values
    scaler.fit_transform(pd.DataFrame(df1[:, 1]))
    vis1 = scaler.inverse_transform(np.array(vis.reshape(-1, 1)))
    y_test1 = scaler.inverse_transform(np.array(y_test.reshape(-1, 1)))
    rmse = np.sqrt(np.mean(np.power((y_test1 - vis1), 2)))
    return rmse

# def F1():
#     dim = 4
#     lb = np.array([12, 12, 8, 0.001])
#     ub = np.array([200, 200, 256, 0.05])
#     x = np.arange(-100, 102, 2)  # x=-100:2:100; y=x;
#     y = x
#     def fobj(x): return f1(x)
#     return {'lb': lb, 'ub': ub, 'dim': dim, 'fobj': fobj}

SearchAgents_no = 30  # Number of search agents
Function_name = "F1"
Max_iteration = 600
dim = 4
lb = np.array([12, 12, 8, 0.001])
ub = np.array([200, 200, 256, 0.05])
# a = ''.join([i for i in Function_name if i.isdigit()])
# ludf = F1(int(a))
# lb, ub, dim, fobj = ludf['lb'], ludf['ub'], ludf['dim'], ludf['fobj']
kq = hmpa.MPA(SearchAgents_no, Max_iteration, lb, ub, dim, f1)
Best_score, Best_pos, Convergence_curve = kq['Top_predator_fit'], kq[
    'Top_predator_pos'], kq['Convergence_curve']
print(Best_score)

neurons1 = int(Best_pos[0])
neurons2 = int(Best_pos[1])
batch_size = int(Best_pos[2])
learn_rate = Best_pos[3]
model = Sequential()
model.add(Bidirectional(LSTM(units=neurons1, return_sequences=True, input_dim=x_train.shape[-1], input_length=x_train.shape[1])))
model.add(Bidirectional(LSTM(units=neurons2)))
model.add(Dropout(learn_rate))
model.add(Dense(units=1))
model.add(Activation("linear"))
model.compile(loss='mean_squared_error', optimizer='Adam')
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
vis = model.predict(x_test)
df1 = df1.values
scaler.fit_transform(pd.DataFrame(df1[:,1]))
vis1 = scaler.inverse_transform(np.array(vis.reshape(-1, 1)))
y_test1 = scaler.inverse_transform(np.array(y_test.reshape(-1, 1)))

mae = np.mean(abs(y_test1 - vis1))
mse = np.mean(np.power((y_test1-vis1), 2))
rmse = np.sqrt(np.mean(np.power((y_test1 - vis1), 2)))
r2 = r2_score(y_test1, vis1)
print('mae:',mae)
print('mse:',mse)
print('rmse:',rmse)
print('r2:',r2)

plt.figure(figsize=(10, 10))
plt.plot(y_test1)
plt.plot(vis1)
plt.title('IMPA_BILSTM real vs pred test')
plt.ylabel('V')
plt.xlabel('X')
plt.legend(['pred', 'real'], loc='lower right')
plt.show()


