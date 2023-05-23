import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

class NN_Network(nn.Module):
    def __init__(self,in_dim,hid,out_dim):
        super(NN_Network, self).__init__()
        self.linear1 = nn.Linear(in_dim,hid)
        self.activate= nn.Sigmoid()
        self.linear2 = nn.Linear(hid,out_dim)


        # self.linear1.weight.data = torch.FloatTensor(0.1*np.ones([2,3]))
        # self.linear1.bias.data = torch.FloatTensor(np.ones([2])*0.1)
        # self.linear2.weight.data =  torch.FloatTensor(np.ones([1,2])*0.1)
        # self.linear2.bias.data =  torch.FloatTensor([0.1])

    def forward(self, input_array):     
        h1 = self.linear1(input_array)
        h2 = self.activate(h1)
        y_pred = self.linear2(h2)
        return y_pred

in_d = 4
hidn = 2
out_d = 3
epochs=10
train_data = pd.read_table("iris/iris-train.txt", header=None)
train_x = torch.FloatTensor(train_data.iloc[:, :-1].values)
train_y = torch.FloatTensor(train_data.iloc[:, -1].values.reshape(-1, 1))
train_y = torch.unsqueeze(train_y, dim=1)
# train_y = torch.nn.functional.one_hot(train_y, num_classes=3)
BPnet = NN_Network(in_d, hidn, out_d)
loss = nn.MSELoss()#定义损失函数
optimizer = torch.optim.Adam(BPnet.parameters(), lr=0.1)#定义优化器

for i in range(epochs):
    pred = BPnet(train_x)
    cost = loss(pred, train_y)#计算损失
    cost.backward()#反向传播
    optimizer.step()#梯度更新
    optimizer.zero_grad()#梯度清零
    print(cost)