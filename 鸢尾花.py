import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

train_data = np.loadtxt("iris/iris-train.txt", delimiter="\t")
X_train = train_data[:, 0:4]
y_train = train_data[:, 4]
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)  # Convert class labels to 0, 1, 2
X_train_hat = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_train = X_train_hat

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    p = softmax(y_pred)
    log_likelihood = -np.log(p[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss

def cross_entropy_gradient(X, y_pred, y_true):
    m = y_true.shape[0]
    p = softmax(y_pred)
    grad = np.dot(X.T, (p - y_true)) / m
    return grad

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.ones(num_iters)
    for i in range(num_iters):
        y_pred = np.dot(X, theta)
        grad = cross_entropy_gradient(X, y_pred, y)
        theta = theta - (alpha * grad)
        J_history[i] = cross_entropy_loss(y_pred, y)
    return theta, J_history

# Set the learning rate and number of iterations
alpha = 0.005
num_iters = 230
theta = np.ones((X_train.shape[1], 3))
# Perform gradient descent to minimize the cost function for each class
for i in range(3):
    y_train_i = (y_train == i).astype(int)
    theta_i, J_history_i = gradient_descent(X_train, y_train_i, theta[:, i], alpha, num_iters)
    theta[:, i] = theta_i
    print(f"Class {i} parameter vector: {theta_i}")
    print(f"Class {i} final cost: {J_history_i[-1]}")

import matplotlib.pyplot as plt
plt.plot(J_history_i)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost function')
plt.show()

test_data = np.loadtxt("iris/iris-test.txt", delimiter="\t")
X_test = test_data[:, 0:4]
X_test_hat = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
X_test = X_test_hat
y_test = test_data[:, 4]
y_test = encoder.transform(y_test)  # Convert class labels to 0, 1, 2
y_pred = np.argmax(np.dot(X_test, theta), axis=1)
y_pred = encoder.inverse_transform(y_pred)  # Convert class labels back to original values
print(classification_report(y_test, y_pred))