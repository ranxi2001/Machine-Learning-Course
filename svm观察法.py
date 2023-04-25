import numpy as np

a = np.array([-3,3,1])
b = np.array([2,4,1])
w0=2/25*a-13/175*b
print(w0)

# w0=np.array([0.39,0.06])
w0=np.array([-0.38857143 , -0.05714286 ])
x_1=np.array([-3,3])
x_2=np.array([2,4])
x_3=np.array([3,2])
x_4=np.array([-5,2])
y=w0@x_1.T+0.00571429
print(y)
y=w0@x_4.T+0.00571429
print(y)
y=w0@x_2.T+0.00571429
print(y)
y=w0@x_3.T+0.00571429
print(y)