from scipy import stats
import numpy as np
Data = np.array([1,2,6,7])

w1 , w2 = 0.5, 0.5
mu1 , mu2 = 1, 5
std1 , std2 = 1, 1

n = len(Data) # 样本长度
zij=np.zeros([n,2])
for t in range(10):
    # E-step 依据当前参数，计算每个数据点属于每个子分布的概率
    z1_up = w1 * stats.norm(mu1 ,std1).pdf(Data)
    z2_up = w2*stats.norm(mu2 , std2).pdf(Data)
    z_all = (w1*stats.norm(mu1 ,std1).pdf(Data)+w2*stats.norm(mu2 ,std2).pdf(Data))+0.001
    rz1 = z1_up/z_all # 为男分布的概率
    rz2 = z2_up/z_all # 为女分布的概率
    # M-step 依据 E-step 的结果，更新每个子分布的参数。
    mu1 = np.sum(rz1*Data)/np.sum(rz1)
    mu2 = np.sum(rz2*Data)/np.sum(rz2)
    std1 = np.sum(rz1*np.square(Data-mu1))/np.sum(rz1)
    std2 = np.sum(rz2*np.square(Data-mu2))/np.sum(rz2)
    w1 = np.sum(rz1)/n
    w2 = np.sum(rz2)/n
for i in range(n):
    zij[i][0] = rz1[i]/(rz1[i]+rz2[i])
    zij[i][1] = rz2[i]/(rz1[i]+rz2[i])

labels = np.argmax(zij, axis=1)
print(labels)