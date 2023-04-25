import numpy as np

train_data = np.loadtxt("iris/iris-train.txt", delimiter="\t")
X_train = train_data[:, 0:4]
y_train = train_data[:, 4]
X_train_hat= np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_train= X_train_hat

def hypothesis(X, theta):
    z = np.dot(X, theta)
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    epsilon = 1e-5  # 防止出现log(0)的情况
    J = (-1 / m) * np.sum(y * np.log(hypothesis(X, theta)+epsilon) + (1 - y) * np.log(1 - hypothesis(X, theta))+epsilon)
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.ones(num_iters)
    for i in range(num_iters):
        theta = theta - (alpha / m) * np.dot(X.T, (hypothesis(X, theta) - y))
        J_history[i] = cost_function(X, y, theta)
    return theta, J_history

# Set the learning rate and number of iterations
alpha = 0.005
num_iters = 230
theta = np.ones(X_train.shape[1])
# Perform gradient descent to minimize the cost function
theta, J_history = gradient_descent(X_train, y_train, theta, alpha, num_iters)

# Print the final parameter vector and cost
print("Parameter vector: ", theta)
print("Final cost: ", J_history[-1])

import matplotlib.pyplot as plt
plt.plot(J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost function')
plt.show()

from sklearn.metrics import classification_report
test_data = np.loadtxt("iris/iris-test.txt", delimiter="\t")
X_test = test_data[:, 0:4]
X_test_hat= np.hstack((np.ones((X_test.shape[0], 1)), X_test))
X_test= X_test_hat
y_test = test_data[:, 4]
y_pred = hypothesis(X_test, theta)


print(classification_report(y_test, y_pred))