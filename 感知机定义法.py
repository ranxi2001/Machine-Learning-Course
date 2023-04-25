import numpy as np
def perceptron(X, Y, max_iter=1000, alpha=1.0):
    """
    X: m x n matrix, m samples with n features
    Y: m x 1 matrix, true labels of m samples
    max_iter: int, maximum training iterations
    alpha: float, learning rate
    """

    # 初始化权重向量和偏置
    w = np.zeros((X.shape[1], 1))#x行1列的0矩阵
    b = 0.0

    # 迭代更新权重和偏置
    for iter in range(max_iter):
        w_last=w
        b_last=b
        for i in range(X.shape[0]):
            y_pred = np.sign(np.dot(X[i], w) + b)
            print("当前参数w={},b={},对于第{}个点".format(w.T,b,i+1))
            print(int(Y[i]*y_pred),Y[i]*(np.dot(X[i], w) + b))
            if y_pred != Y[i]:
                w = w + alpha * Y[i] * X[i].reshape(-1, 1)
                b = b + alpha * Y[i]
                break
        if (np.array_equal(w, w_last)==False and b!=b_last):
            w_last= w
            b_last = b
        elif (np.array_equal(w, w_last) and b==b_last):
            break
    return w, b
def main():
    X = np.array(
        [
            [-3, 3],
            [-5, 2],
            [2 , 4],
            [3 , 2]
        ]
    )
    Y=np.array([1,1,-1,-1])
    perceptron(X, Y, max_iter=100, alpha=0.1)
if __name__ == '__main__':
    main()