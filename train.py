import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

torch.manual_seed(1)

def train(model, epoch=10000, low=3, high=20, lr=0.01, mom=0.9, batch=24):
    EPOCH = epoch
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
    losses = []
    gen_losses = []
    for i in range(EPOCH):
        model.train()
        model.zero_grad()
        n = np.random.randint(low, high, size=1)[0]

        data, label = sampler(length=n, batch=batch)
        pred = model(data)
        loss = crit(pred, label)
        add = 0
        if 'loss' in model.__dict__:
            add = model.loss
            #print(add)
        
        #print(i, loss.data[0])

        factor = 0.5 ** ( i // 2000)
        ((loss + add) * factor).backward()
        optimizer.step()
        ##testing
        if i % 100 == 0:
            model.eval()
            data, label = sampler(length=high+5, batch=50)
            pred = model(data)
            #print(i, loss.data[0])
            gen_loss = crit(pred, label)
            losses.append(loss.data.cpu().numpy()[0])
            gen_losses.append(gen_loss.data.cpu().numpy()[0])
            if VERBOSE:
                print(i, loss.data.cpu().numpy()[0], gen_loss.data.cpu().numpy()[0])

    return [losses, gen_losses]

def test(model, high=60):
    length_losses = []
    total = 0
    for l in range(1, high):
        model.eval()
        data, label = sampler(length=l, batch=100)
        pred = model(data)
        #print(i, loss.data[0])
        loss = crit(pred, label)
        length_losses.append(loss.data.cpu().numpy())
        total = total + loss.data.cpu().numpy()
    print(total)
    return length_losses


