import numpy as np
train_data = np.loadtxt("./iris/iris-train.txt", delimiter="\t")

def hypothesis(X, theta):
    return np.dot(X, theta)

def cost_function(X, y, theta):
    m = len(y)
    J = np.sum((hypothesis(X, theta) - y) ** 2) / (2 * m)
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        theta = theta - (alpha / m) * np.dot(X.T, (hypothesis(X, theta) - y))
        J_history[i] = cost_function(X, y, theta)
    return theta, J_history

# Initialize the parameter vector
X_train = train_data[:, 0:4]
y_train = train_data[:, 4]
X_train_hat= np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_train= X_train_hat
theta = np.ones(X_train.shape[1])

# Set the learning rate and number of iterations
alpha = 0.0005
num_iters = 120

# Perform gradient descent to minimize the cost function
theta, J_history = gradient_descent(X_train, y_train, theta, alpha, num_iters)

# Print the final parameter vector and cost
print("Parameter vector: ", theta)
print("Final cost: ", J_history[-1])

# Plot the cost function
import matplotlib.pyplot as plt
plt.plot(J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost function')
plt.show()