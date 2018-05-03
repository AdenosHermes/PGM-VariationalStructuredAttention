
from torch.autograd import Variable
import torch 
torch.manual_seed(1)
#VERBOSE = True

import numpy as np

def chunk_samp(length=10, batch=2):
    x = np.random.randint(0, 2, size=(batch, length))
                          
    y = x
    y1 = np.concatenate((y, np.zeros((batch, 1))), axis=1)
    y2 = np.concatenate((np.zeros((batch, 1)), y), axis=1)
    #print(y)
    #print(x, y)
    z = y1 - y2
    count = (z == 1).sum(axis=1)
    x = Variable(torch.Tensor(x)).cuda()
    count = Variable(torch.Tensor(count)).cuda()
    return x, count
    


def sum_samp(length=10, batch = 2):
    x = np.random.randint(1, 10, size=(batch, length))
#     x = np.random.randint(0, 2, size=(batch, length))
    mask = np.random.randint(0, 2, size=(batch, length))
    maker = - np.ones((batch,2))

    y = np.sum(x * mask, axis = 1)
    x = np.concatenate((x, maker), axis=1)
    x = np.concatenate((x, mask), axis=1)
    
    x = Variable(torch.Tensor(x)).cuda()
    y = Variable(torch.Tensor(y)).cuda()
    
    return x, y
    
