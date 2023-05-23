import pandas
import numpy as np
from sklearn.metrics import classification_report
# 导入txt数据
iris_train = pandas.read_table("iris/iris-train.txt", header=None)
iris_train.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
iris_test = pandas.read_table("iris/iris-test.txt", header=None)
iris_test.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

#打乱顺序
array = iris_train.values#
np.random.seed(1377)
np.random.shuffle(array)

#独热编码
def onehot(targets, num_out):
    onehot = np.zeros((num_out, targets.shape[0]))
    for idx, val in enumerate(targets.astype(int)):
        onehot[val, idx] = 1.
    return onehot.T

#生成一个矩阵，大小为m*n,并且设置默认零矩阵
def makematrix(m, n, fill=0.0):
    X_train = []
    for i in range(m):
        X_train.append([fill] * n)
    return X_train


#函数sigmoid()
def sigmoid(x):
    a = 1 / (1 + np.exp(-x))
    return a

#函数sigmoid()的导数
def derived_sigmoid(x):
    return x * (1 - x)
    # return 1.0 - x ** 2

#构造四层BP网络架构
class BPNN:
    def __init__(self, num_in, num_hidden1, num_hidden2, num_out):
        # 输入层，隐藏层，输出层的节点数
        self.num_in = num_in + 1  # 增加一个偏置结点 4
        self.num_hidden1 = num_hidden1 + 1  # 增加一个偏置结点 4
        self.num_hidden2 = num_hidden2 + 1
        self.num_out = num_out

        # 激活神经网络的所有节点
        self.active_in = [1.0] * self.num_in
        self.active_hidden1 = [1.0] * self.num_hidden1
        self.active_hidden2 = [1.0] * self.num_hidden2
        self.active_out = [1.0] * self.num_out

        # 创建权重矩阵
        self.wight_in = makematrix(self.num_in, self.num_hidden1)
        self.wight_h1h2 = makematrix(self.num_hidden1, self.num_hidden2)
        self.wight_out = makematrix(self.num_hidden2, self.num_out)

        # 对权值矩阵赋初值
        for i in range(self.num_in):
            for j in range(self.num_hidden1):
                self.wight_in[i][j] = np.random.normal(0.0, pow(self.num_hidden1, -0.5))  # 输出num_in行,num_hidden列权重矩阵，随机生成满足正态分布的权重
        for i in range(self.num_hidden1):
            for j in range(self.num_hidden2):
                self.wight_h1h2[i][j] = np.random.normal(0.0, pow(self.num_hidden2, -0.5))
        for i in range(self.num_hidden2):
            for j in range(self.num_out):
                self.wight_out[i][j] = np.random.normal(0.0, pow(self.num_out, -0.5))

        # 最后建立动量因子（矩阵）
        self.ci = makematrix(self.num_in, self.num_hidden1)
        self.ch1h2 = makematrix(self.num_hidden1, self.num_hidden2)
        self.co = makematrix(self.num_hidden2, self.num_out)


        # 信号正向传播
    def update(self, inputs):
        a=len(inputs)
        if len(inputs) != self.num_in - 1:
            raise ValueError('与输入层节点数不符')

        # 数据输入输入层
        for i in range(self.num_in - 1):
            # self.active_in[i] = sigmoid(inputs[i])  #或者先在输入层进行数据处理
            self.active_in[i] = inputs[i]  # active_in[]是输入数据的矩阵

        # 数据在隐藏层1的处理
        for i in range(self.num_hidden1):
            sum = 0.0
            for j in range(self.num_in):
                sum = sum + self.active_in[j] * self.wight_in[j][i]
            self.active_hidden1[i] = sigmoid(sum)  # active_hidden[]是处理完输入数据之后存储，作为输出层的输入数据

        # 数据在隐藏层2的处理
        for i in range(self.num_hidden2):
            sum = 0.0
            for j in range(self.num_hidden1):
                sum = sum + self.active_hidden1[j] * self.wight_h1h2[j][i]
            self.active_hidden2[i] = sigmoid(sum)  # active_hidden[]是处理完输入数据之后存储，作为输出层的输入数据

        # 数据在输出层的处理
        for i in range(self.num_out):
            sum = 0.0
            for j in range(self.num_hidden2):
                sum = sum + self.active_hidden2[j] * self.wight_out[j][i]
            self.active_out[i] = sigmoid(sum)  # 与上同理

        return self.active_out[:]

    # 误差反向传播
    def errorbackpropagate(self, targets, lr, m):  # lr是学习率， m是动量因子
        if len(targets) != self.num_out:
            raise ValueError('与输出层节点数不符！')

        # 首先计算输出层的误差
        out_deltas = [0.0] * self.num_out
        for i in range(self.num_out):
            error = targets[i] - self.active_out[i]
            out_deltas[i] = derived_sigmoid(self.active_out[i]) * error

        # 计算隐藏层2的误差
        hidden2_deltas = [0.0] * self.num_hidden2
        for i in range(self.num_hidden2):
            error = 0.0
            for j in range(self.num_out):
                error = error + out_deltas[j] * self.wight_out[i][j]
            hidden2_deltas[i] = derived_sigmoid(self.active_hidden2[i]) * error

        # 计算隐藏层1的误差
        hidden1_deltas = [0.0] * self.num_hidden1
        for i in range(self.num_hidden1):
            error = 0.0
            for j in range(self.num_hidden2):
                error = error + hidden2_deltas[j] * self.wight_h1h2[i][j]
            hidden1_deltas[i] = derived_sigmoid(self.active_hidden1[i]) * error

        # 更新输出层权值
        for i in range(self.num_hidden2):
            for j in range(self.num_out):
                change = out_deltas[j] * self.active_hidden2[i]
                self.wight_out[i][j] = self.wight_out[i][j] + lr * change + m * self.co[i][j]
                self.co[i][j] = change

        # 更新隐藏层间权值
        for i in range(self.num_hidden1):
            for j in range(self.num_hidden2):
                change = hidden2_deltas[j] * self.active_hidden1[i]
                self.wight_h1h2[i][j] = self.wight_h1h2[i][j] + lr * change + m * self.ch1h2[i][j]
                self.ch1h2[i][j] = change

        # 然后更新输入层权值
        for i in range(self.num_in):
            for j in range(self.num_hidden1):
                change = hidden1_deltas[j] * self.active_in[i]
                self.wight_in[i][j] = self.wight_in[i][j] + lr * change + m * self.ci[i][j]
                self.ci[i][j] = change

        # 计算总误差
        error = 0.0
        for i in range(self.num_out):
            error = error + 0.5 * (targets[i] - self.active_out[i]) ** 2
        return error

    # 测试
    def test(self, X_test):
        for i in range(X_test.shape[0]):
            print(X_test[i, 0:4], '->', self.update(X_test[i, 0:4]))

    # 权重
    def weights(self):
        print("输入层权重")
        for i in range(self.num_in):
            print(self.wight_in[i])
        print("输出层权重")
        for i in range(self.num_hidden2):
            print(self.wight_out[i])

    def train(self, train, itera=100, lr=0.1, m=0.1):
        for i in range(itera):
            error = 0.0
            for j in range(100):#训练集的大小
                inputs = train[j, 0:4]
                d = onehot(train[:,4], self.num_out)
                targets = d[j, :]
                self.update(inputs)
                error = error + self.errorbackpropagate(targets, lr, m)
            if i % 100 == 0:
                print('误差 %-.5f' % error)

    def show_accuracy(self, X, Y):
        count=0
        Y_pred=[]
        for i in range(X.shape[0]):
            h=self.update(X[i, 0:4])
            y_pred=np.argmax(h)
            Y_pred.append(y_pred)
            if y_pred==Y[i]:
                count+=1
        print("准确率为：",count/X.shape[0])
        print(count)
        print(classification_report(Y, Y_pred))

# 实例
def Mytrain(train,X_test, Y_test):
    # 创建神经网络，4个输入节点，10个隐藏层1节点，6个隐藏层2节点,3个输出层节点
    n = BPNN(4, 10, 6, 3)
    # 训练神经网络
    print("start training\n--------------------")
    n.train(train,itera=1000)
    n.weights()
    # n.test(X_test)
    n.show_accuracy(X_test, Y_test)


if __name__ == '__main__':
    train = array[:, :]  # 训练集
    X_test = iris_test.values[:, :]  # 测试集
    Y_test = iris_test.values[:, 4]  # 测试集的标签
    Mytrain(train, X_test, Y_test)