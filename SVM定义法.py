import numpy as np

def svm_classify_array(data_arr, label_arr, eta=0.1):
    w = np.zeros(data_arr.shape[1])
    b = 0
    m = len(data_arr)
    for epoch in range(1000):
        for i in range(m):
            judge = label_arr[i] * (np.dot(w, data_arr[i]) + b)
            if judge * label_arr[i] < 1:
                w = w + eta * (label_arr[i] * data_arr[i] - 2 * 1 / (epoch + 1) * w)
                b = b + eta * label_arr[i]
                bias = -np.sum(w * label_arr) / len(label_arr)
    return w, bias

input_vecs = [[-3, 3], [-5, 2], [2, 4], [3, 2]]
input_labels = [[1], [1], [-1], [-1]]
input_vecs = np.array(input_vecs)
input_labels = np.array(input_labels)
weight, bias = svm_classify_array(input_vecs, input_labels)
print('weight=', weight)
print('bias=', bias)