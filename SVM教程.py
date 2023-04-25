import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, learning_rate=0.001,lambda_param=0.0001,n_iters=10000):
        self.a=learning_rate
        self.lambda_param=lambda_param
        self.epoch=n_iters
        self.w=None
        self.b=None

    def fit(self,X,y):
        n_samples,n_features=X.shape
        y_=np.where(y<=0,-1,1)

        self.w=np.zeros(n_features)
        self.b=0
        for epoch in range(self.epoch):
            for idx,x_i in enumerate(X):
                condition=y_[idx]*(np.dot(x_i,self.w)-self.b)>=1
                if condition:
                    self.w=self.w-self.a*(2*self.lambda_param*self.w)
                else:
                    self.w=self.w-self.a*(2*self.lambda_param*self.w-np.dot(x_i,y_[idx]))
                    self.b=self.b-self.a*y_[idx]

    def predict(self,X):
        linear_output=np.dot(X,self.w)-self.b
        return np.sign(linear_output)
#模拟数据
# X,y=datasets.make_blobs(n_samples=50,n_features=2,centers=2,cluster_std=1.05,random_state=40)
# y=np.where(y==0,-1,1)
#题目数据
X = np.array([[-3, 3], [-5, 2], [2, 4], [3, 2]])
y = np.array([1, 1, -1, -1])
clf=SVM()
clf.fit(X,y)
print(clf.w,clf.b)

def visualize_svm():
    def get_hyperplane_value(x,w,b,v):
        return (-w[0]*x-b+v)/w[1]
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.scatter(X[:,0],X[:,1],marker='o',c=y)
    # x0_1=np.amin(X[:,0])
    # x0_2=np.amax(X[:,0])
    x0_1=-6
    x0_2=6
    x1_1=get_hyperplane_value(x0_1,clf.w,clf.b,0)
    x1_2=get_hyperplane_value(x0_2,clf.w,clf.b,0)
    x1_1_m=get_hyperplane_value(x0_1,clf.w,clf.b,-1)
    x1_2_m=get_hyperplane_value(x0_2,clf.w,clf.b,-1)
    x1_1_p=get_hyperplane_value(x0_1,clf.w,clf.b,1)
    x1_2_p=get_hyperplane_value(x0_2,clf.w,clf.b,1)
    ax.plot([x0_1,x0_2],[x1_1,x1_2],'r--')
    ax.plot([x0_1,x0_2],[x1_1_m,x1_2_m],'b')
    ax.plot([x0_1,x0_2],[x1_1_p,x1_2_p],'b')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle='--')
    # x1_min=np.amin(X[:,1])
    # x1_max=np.amax(X[:,1])
    # ax.set_ylim([x1_min-3,x1_max+3])
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.show()

visualize_svm()