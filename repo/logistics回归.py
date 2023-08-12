import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
# 生成示例数据
# 假设有三个特征变量 X1、X2、X3 和一个多分类目标变量 Y (0、1、2)
data=pd.read_excel('F:\code\pythoncode\数据1.xlsx')
print(data)

X = data.keys()[1:9]
print(X)
data1=data[X].values
print(data1)

Y = data.keys()[-1]
print(Y)
data2=data[Y].values
print(data2)

# 将数据集分为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(data1, data2, test_size=0.2, random_state=42)

# 创建逻辑回归模型
model = LogisticRegression(multi_class='auto', solver='lbfgs')

# 拟合模型
model.fit(X_train, Y_train)

# 进行预测
Y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(Y_test, Y_pred)
print("准确率:", accuracy)