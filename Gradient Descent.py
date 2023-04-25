import numpy as np
def computeCost(X, theta, y):
    return 0.5 * np.mean(np.square(h(X, theta) - y))
def h(X, theta):
    return np.dot(X, theta)
def gradientDescent(X, theta , y, iterations , alpha):
    Cost = []
    Cost.append(computeCost(X, theta , y))
    for i in range(iterations):
        grad0 = np.mean(h(X, theta) - y)
        grad1 = np.mean((h(X, theta) - y) * (X[:,1].
        reshape([len(X), 1])
        ))
        theta[0] = theta[0] - alpha * grad0
        theta[1] = theta[1] - alpha * grad1
        Cost.append(computeCost(X, theta , y))
    return theta , Cost
theta = np.zeros ((1,1))
iterations = 100
alpha = 0.05