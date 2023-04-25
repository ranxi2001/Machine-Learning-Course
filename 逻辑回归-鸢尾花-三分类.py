import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'labels']
    data = np.array(df.iloc[:, [0, 1, 2, 3, -1]])  # 共149个样本，四个特征，3种标签
    # print(df) 想看数据集长相的可以输出看一下
    return data[:, :4], data[:, -1]  # 取前四列作为四个特征，第五列是标签


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


class LogisticRegressionClassifier:
    def __init__(self, max_iter=200, lr=0.01):
        self.max_iter = max_iter  # 最大迭代次数
        self.lr = lr  # 学习率

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))  # 这里必须用np.exp，python库自带的exp无法对矩阵中的元素进行计算

    def data_matrix(self, X):  # 在数据集中增加一个偏置项，等于让矩阵多一个维度
        data_mat = []
        for x in X:
            data_mat.append([1, *x])
        return data_mat

    def softmax(self, d):  # 多分类用 softmax
        return np.exp(d) / np.sum(np.exp(d))

    def fit(self, X, y):
        data_mat = self.data_matrix(X)
        self.weights = np.zeros((len(data_mat[0]), 3), dtype=np.float32)
        # 初始化参数，维度是(特征数，类别数) = (4x3)

        for step_ in range(self.max_iter):
            for i in range(len(data_mat)):
                # pre = self.sigmoid(np.dot(data_mat[i], self.weights))
                pre = self.softmax(np.dot(data_mat[i], self.weights))
                # (1,3) = (1,4) x (4,3)
                obj = np.eye(3)[int(y[i])]  # 这里是将标签值，变成独热向量, 如[1] 变成 [0 1 0]
                err = pre - obj
                self.weights -= self.lr * np.transpose([data_mat[i]]) * err
                # (4,3) = (1,1) x (4,1) x (1,3)
            if (step_ % 5 == 0):
                print("*********************************************************")
                print("round {}\nweights\n {} \nerr {} \nscore {}".format(step_, self.weights, err,
                                                                          self.score(X_test, y_test)))
                print("distribution\t", pre)

    def score(self, X, y):
        X = self.data_matrix(X)
        right = 0
        for i in range(len(X)):
            pre = np.dot(X[i], self.weights)
            # (1,3) = (1,4) x (4,3)
            pre2 = np.argmax(pre)  # 找到(1,3)这个向量中值最大对应的索引，也就是预测的类别
            if pre2 == y[i]:  # 索引跟真实值一样，说明预测正确
                right += 1
        return right / len(X)


lrc = LogisticRegressionClassifier(max_iter=500)
lrc.fit(X_train, y_train)
# print(lrc.score(X_test, y_test))
