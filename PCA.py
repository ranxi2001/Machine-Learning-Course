import numpy as np

X=np.array([
    [ 3.25, 1.85, -1.29],
    [ 3.06, 1.25, -0.18],
    [ 3.46, 2.68, 0.64],
    [ 0.3 , -0.1 , -0.79],
    [ 0.83, -0.21, -0.88],
    [ 1.82, 0.99, 0.16],
    [ 2.78, 1.75, 0.51],
    [ 2.08, 1.5 , -1.06],
    [ 2.62, 1.23, 0.04],
    [ 0.83, -0.69, -0.61]])

def pca(X, d):
    # Centralization中心化
    means = np.mean(X, 0)
    X = X - means
    print(X)
    # Covariance Matrix协方差矩阵
    M=len(X)
    X=np.mat(X)
    #covM1 =(X.T * X)/(M-1)
    covM2=np.cov(X.T)
    eigval , eigvec = np.linalg.eig(covM2)
    indexes = np.argsort(eigval)[-d:]
    W = eigvec[:, indexes]
    return X*W
print(pca(X, 2))