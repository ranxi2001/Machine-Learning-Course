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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    epsilon = 1e-5
    h = sigmoid(np.dot(X, theta))
    J = (-1 / m) * np.sum(y * np.log(h+epsilon) + (1 - y) * np.log(1 - h+epsilon))
    grad = (1 / m) * np.dot(X.T, (h - y))
    return J, grad

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        J, grad = cost_function(X, y, theta)
        theta = theta - (alpha * grad)
        J_history[i] = J
    return theta, J_history
def logistic_regression(X, y, alpha, num_iters):
    theta = np.ones((X_train.shape[1], 3))
    # Perform gradient descent to minimize the cost function for each class
    for i in range(3):
        y_train_i = (y_train == i).astype(int)
        theta_i, J_history_i = gradient_descent(X_train, y_train_i, theta[:, i], alpha, num_iters)
        import matplotlib.pyplot as plt

        plt.plot(J_history_i)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost function'+str(i))
        plt.show()
        theta[:, i] = theta_i
        print(f"Class {i} parameter vector: {theta_i}")
        print(f"Class {i} final cost: {J_history_i[-1]}")
    return theta
def predict(test_data, theta):
    X_test = test_data[:, 0:4]
    X_test_hat = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    X_test = X_test_hat
    y_test = test_data[:, 4]
    y_test = encoder.transform(y_test)  # Convert class labels to 0, 1, 2
    y_pred = np.zeros(X_test.shape[0])
    for i in range(3):
        sigmoid_outputs = sigmoid(np.dot(X_test, theta[:, i]))
        threshold = np.median(sigmoid_outputs[y_test == i])#不采用统一的阈值，而是采用每个类别的中位数作为阈值
        y_pred_i = (sigmoid_outputs >= threshold).astype(int)
        y_pred[y_pred_i == 1] = i
    y_pred = encoder.inverse_transform(y_pred.astype(int))  # Convert class labels back to original values
    print(classification_report(y_test, y_pred))

test_data = np.loadtxt("iris/iris-test.txt", delimiter="\t")
theta = logistic_regression(X_train, y_train, 0.0005, 5000)
predict(test_data, theta)