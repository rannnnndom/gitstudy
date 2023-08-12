import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = np.array([71.1,72.4,72.4,72.1,71.4]) #噪音数据
lens = len(data)

# 数据检验
## 计算级比
lambds = []
for i in range(1, lens):
    lambds.append(data[i-1]/data[i])
## 计算区间
X_min = np.e**(-2/(lens+1))
X_max = np.e**(2/(lens+1))
## 检验
test = True
for lambd in lambds:
    if (lambd < X_min or lambd > X_max):
        test = False
if test == False:
        print('该数据不可以用灰色GM(1,1)模型')
else:
        print('该数据可以用灰色GM(1,1)模型')

# 构建灰色模型GM(1,1)
## 生成累加数列
data_add = []
for i in range(1, lens):
    data_add = data.cumsum()
## 生成紧邻均值序列
ds = []
zs = []
for i in range(1, lens):
    ds.append(data[i])
    zs.append(-1/2*(data_add[i-1]+data_add[i]))
## 生成矩阵和最小二乘法
B = np.array(zs).reshape(lens-1,1)
one = np.ones(lens-1)
B = np.c_[B, one]  # 加上一列1
Y = np.array(ds).reshape(lens-1,1)
a, b = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
print('a='+str(a))
print('b='+str(b))

# 预测
def forecast(k):
    c = b/a
    return (data[0]-c)*(np.e**(-a*k))+c
data_1_for = []  # 累加预测值
data_0_for = []  # 原始预测值
data_1_for.append(forecast(0))
data_0_for.append(data_1_for[0])
for i in range(1, lens+5):  # 多预测5次
    data_1_for.append(forecast(i))
    data_0_for.append(data_1_for[i]-data_1_for[i-1])
print('预测值为：')
for i in data_0_for:
    print(i)

# 模型检验
## 预测结果方差
data_h = np.array(data_0_for[0:lens]).T
sum_h = data_h.sum()
mean_h = sum_h/lens
S1 = np.sum((data_h-mean_h)**2)/lens
## 残差方差
e = data - data_h
e_sum = e.sum()
e_mean = e_sum/lens
S2 = np.sum((e-e_mean)**2)/lens
## 后验差比
C = S2/S1
## 结果
if (C <= 0.35):
    print('1级，效果好')
elif (C <= 0.5 and C >= 0.35):
    print('2级，效果合格')
elif (C <= 0.65 and C >= 0.5):
    print('3级，效果勉强')
else:
    print('4级，效果不合格')

# 可视化
plt.figure(figsize=(30, 30), dpi=100)
x1 = np.linspace(1, 5, 5)
x2 = np.linspace(1, 10, 10)
plt.subplot(121)
plt.title('x^0')
plt.plot(x2, data_0_for, 'r--', marker='*')
plt.scatter(x1, data, marker='^')
plt.subplot(122)
plt.title('x^1')
plt.plot(x2, data_1_for, 'r--', marker='*')
plt.scatter(x1, data_add, marker='^')
plt.show()

#Process finished with exit code 0
