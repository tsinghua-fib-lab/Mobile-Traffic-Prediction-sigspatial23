import torch
import numpy as np
import pandas as pd

# produce data slices for training and testing

def data_transform(data,n_his,n_pred):
    # produce data slices for training and testing
    n_route = data.shape[0]
    l = data.shape[1]
    num = l-n_his-n_pred
    x = np.zeros([num,n_route,n_his])
    y = np.zeros([num,n_route,1])


    cnt = 0
    for i in range(num):
        head = i
        tail = i+n_his
        his = data[:,head:tail]
        x[cnt,:,:] = data[:,head:tail].reshape(n_route,n_his)
        y[cnt] = data[:,tail+n_pred-1].reshape(n_route,1)
        cnt += 1
    x = x.reshape(num,n_route,n_his,1)
    return torch.Tensor(x), torch.Tensor(y)
