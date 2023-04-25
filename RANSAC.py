import random
import numpy as np
from matplotlib import pyplot as plt


def ransac_line_fit(data, n_iterations, threshold):
    """
    RANSAC algorithm for linear regression.

    Parameters:
    data (list): list of tuples representing the data points
    n_iterations (int): number of iterations to run RANSAC
    threshold (float): maximum distance a point can be from the line to be considered an inlier

    Returns:
    tuple: slope and y-intercept of the best fit line
    """
    best_slope, best_intercept = None, None
    best_inliers = []

    for i in range(n_iterations):
        # Randomly select two points from the data
        sample = random.sample(data, 2)
        x1, y1 = sample[0]
        x2, y2 = sample[1]

        # Calculate slope and y-intercept of line connecting the two points
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Find inliers within threshold distance of the line
        inliers = []
        outliers = []
        for point in data:
            x, y = point
            distance = abs(y - (slope * x + intercept))
            distance = distance / np.sqrt(slope ** 2 + 1)
            if distance <= threshold:
                inliers.append(point)
            else:
                outliers.append(point)

        # If the number of inliers is greater than the current best, update the best fit line
        if len(inliers) > len(best_inliers):
            best_slope = slope
            best_intercept = intercept
            best_inliers = inliers

    outliers = [point for point in data if point not in best_inliers]
    # Plot the data points, best fit line, and inliers and outliers
    fig, ax = plt.subplots()
    # ax.scatter([x for x, y in data], [y for x, y in data], color='black')
    ax.scatter([x for x, y in best_inliers], [y for x, y in best_inliers], color='green')
    ax.scatter([x for x, y in outliers], [y for x, y in outliers], color='black')
    x_vals = np.array([-5,5])
    y_vals = best_slope * x_vals + best_intercept
    ax.plot(x_vals, y_vals, '-', color='red')
    # threshold_line = best_slope * x_vals + best_intercept + threshold*np.sqrt((1/best_slope) ** 2 + 1)
    threshold_line = best_slope * x_vals + best_intercept + threshold * np.sqrt(best_slope ** 2 + 1)
    ax.plot(x_vals, threshold_line, '--', color='blue')
    # threshold_line = best_slope * x_vals + best_intercept - threshold*np.sqrt((1/best_slope) ** 2 + 1)
    threshold_line = best_slope * x_vals + best_intercept - threshold * np.sqrt(best_slope ** 2 + 1)
    ax.plot(x_vals, threshold_line, '--', color='blue')
    # ax.set_xlim([-10, 10])
    ax.set_ylim([-6, 6])
    plt.show()

    return best_slope, best_intercept


import numpy as np

# Generate 10 random points with x values between 0 and 10 and y values between -5 and 5
data = [(x, y) for x, y in zip(np.random.uniform(-5, 5, 10), np.random.uniform(-5, 5, 10))]

print(data)

# Fit a line to the data using RANSAC
slope, intercept = ransac_line_fit(data, 10000, 1)
print(slope, intercept)
