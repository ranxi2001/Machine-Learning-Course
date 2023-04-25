import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 坐标点数据
X = np.array([[-3, 3], [-5, 2], [2, 4], [3, 2]])
y = np.array([1, 1, -1, -1])

# 定义SVM分类器
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# 定义SVM超平面
w = clf.coef_[0]
b = clf.intercept_
x1 = np.arange(-6, 6)
x2 = -(w[0] * x1 + b) / w[1]

# 创建一个散点图
fig, ax = plt.subplots()

# 设置坐标轴范围和中心点
x_min, x_max = -6, 6
y_min, y_max = -6, 6
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 绘制四个点
ax.scatter(X[0, 0], X[0, 1], color='r')
ax.scatter(X[1, 0], X[1, 1], color='r')
ax.scatter(X[2, 0], X[2, 1], color='b')
ax.scatter(X[3, 0], X[3, 1], color='b')

# 绘制SVM超平面
ax.plot(x1, x2, c='r',label='svm')

# 在散点周围添加坐标注释
ax.annotate(f'({X[0, 0]}, {X[0, 1]})', xy=(X[0, 0], X[0, 1]), xytext=(5, 5), textcoords='offset points')
ax.annotate(f'({X[1, 0]}, {X[1, 1]})', xy=(X[1, 0], X[1, 1]), xytext=(5, 5), textcoords='offset points')
ax.annotate(f'({X[2, 0]}, {X[2, 1]})', xy=(X[2, 0], X[2, 1]), xytext=(5, 5), textcoords='offset points')
ax.annotate(f'({X[3, 0]}, {X[3, 1]})', xy=(X[3, 0], X[3, 1]), xytext=(5, 5), textcoords='offset points')

# 增加图例 显示图像
plt.legend(loc='best')
plt.show()
