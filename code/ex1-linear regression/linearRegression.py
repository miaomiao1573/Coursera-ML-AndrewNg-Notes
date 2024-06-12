import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return (np.sum(inner) / (2 * len(X)))


path =  'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# data.head()
#散点图
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
# plt.show()

#添加一行，使用向量化方案来处理数据
data.insert(0, 'Ones', 1)
#变量初始化
# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]#X是所有行，去掉最后一列
y = data.iloc[:,cols-1:cols]#y是所有行，最后一列
print(X.head())
#矩阵化
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

u = computeCost(X, y, theta)
print(u)

def gradientDescent(X, y, theta, alpha, iters):
    #初始化各种列表
    temp = np.matrix(np.zeros(theta.shape))
    paramter = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    #
    for i in range(iters):
        error= (X * theta.T)-y

        for j in range(paramter):
            term = np.multiply(error,X[:,j])
            #看看四周的下降梯度
            temp[:j] = theta[0,j]-((alpha / len(X)) * np.sum(term))

        theta = temp
        #计算方差，每一步的花费
        cost = computeCost(X, y, theta)
    return theta, cost
#赋值步数和下降函数
alpha = 0.01
iters = 1000
#获取
g, cost = gradientDescent(X, y, theta, alpha, iters)

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0,0]+(g[0,1] * x)
fig, ax= plt.subplots(figsize = (12,8))
ax.plot(x, f, 'r', label='Predicted')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
