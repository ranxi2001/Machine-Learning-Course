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

# 绘制散点图和SVM超平面
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.plot(x1, x2, c='r',label='svm')

# 坐标轴固定在图像中央
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.axhline(y=0, color='gray', linestyle='--')
plt.axvline(x=0, color='gray', linestyle='--')

# 增加图例
plt.legend(loc='best')
plt.show()
print(w,b)