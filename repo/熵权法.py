#导入相关库
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#读取数据
data=pd.read_excel('F:\code\pythoncode\数据.xlsx')
print(data)
 
label_need=data.keys()[1:]
print(label_need)
data1=data[label_need].values
print(data1)
 
#指标正向    化处理后数据为data2
data2=data1
print(data2)
 
#越小越优指标位置,注意python是从0开始计数，对应位置也要相应减1
index=[1,2] 
for i in range(0,len(index)):
    data2[:,index[i]]=max(data1[:,index[i]])-data1[:,index[i]]
print(data2)
 
#0.002~1区间归一化
[m,n]=data2.shape
data3=copy.deepcopy(data2)
ymin=0.002
ymax=1
for j in range(0,n):
    d_max=max(data2[:,j])
    d_min=min(data2[:,j])
    data3[:,j]=(ymax-ymin)*(data2[:,j]-d_min)/(d_max-d_min)+ymin
print(data3)
 
#计算信息熵
p=copy.deepcopy(data3)
for j in range(0,n):
    p[:,j]=data3[:,j]/sum(data3[:,j])
print(p)
E=copy.deepcopy(data3[0,:])
for j in range(0,n):
    E[j]=-1/np.log(m)*sum(p[:,j]*np.log(p[:,j]))
print(E)
 
# 计算权重
w=(1-E)/sum(1-E)
print(w)

# 依据给定权重分类
# 计算加权得分
weighted_score = np.dot(data3, w)
# 根据得分将数据分为四个类别
classes = ['差', '中', '良', '优']
classified_data = np.digitize(weighted_score, [0.25, 0.5, 0.75, 1.0])
data['分类结果'] = classified_data
# 打印分类结果
print(data)

#可视化分类结果

# 统计每个类别的数量
class_counts = data['分类结果'].value_counts()

# 定义类别标签和颜色
class_labels = ['bad', 'mid', 'good', 'excellent']
colors = ['red', 'orange', 'green', 'blue']
# 绘制条形图
plt.bar(class_labels, class_counts, color=colors)
# 添加标题和标签
plt.title('Data classification results')
plt.xlabel('category')
plt.ylabel('amount')

# 显示图形
plt.show()

# 指定输出的 Excel 文件名
output_file = 'classified_data.xlsx'
# 将数据输出到 Excel 文件
data.to_excel(output_file, index=False)

print(f"已将分类结果保存到 {output_file}")