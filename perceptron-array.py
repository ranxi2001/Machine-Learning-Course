import numpy as np
def percep_classify_array(data_arr,label_arr,eta=0.1):
    w=np.array([0,0])
    b=0
    m=len(data_arr)
    error_data = True
    while error_data:               #利用收敛特性不采用迭代方式进行，采用while循环
        error_data = False
        for i in range(m):
            judge = label_arr[i]*(np.dot(w,data_arr[i])+b)
            if judge <=0:
                error_data=True
                w=w+eta*(label_arr[i]*data_arr[i])
                b=b+eta*label_arr[i]
                print('w=',w,'b=',b,'误分类点：x_'+str(i+1))

    return w,b
input_vecs = [[-3,3],[-5,2],[2,4],[3,2]]
input_labels = [[1],[1],[-1],[-1]]
input_vecs = np.array(input_vecs)
input_labels = np.array(input_labels)
weight,bias = percep_classify_array(input_vecs,input_labels)
print('weight=',weight,'bias=',bias,'没有误分类点了')