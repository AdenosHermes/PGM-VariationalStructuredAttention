import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

torch.manual_seed(1)


class BiLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, out_dim=1):
        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim //2,
                            num_layers=1, bidirectional=True)

        self.hidden = self.init_hidden()
        
        self.hid2out = nn.Linear(hidden_dim, out_dim)
        
        
    def init_hidden(self, batch=1):
        return (Variable(torch.randn(2, batch, self.hidden_dim // 2)).cuda(), 
                Variable(torch.randn(2, batch, self.hidden_dim // 2)).cuda())
    

    def forward(self, sentence):
        batch_size = sentence.shape[0]
        time_dim = sentence.shape[1]
        hidden = self.init_hidden(batch_size)
        
        x = torch.transpose(sentence, 1, 0)
        x = x.unsqueeze(2)
        
        #print(x.shape)
        #print(hidden[0].shape)
        
        #for i in range(time_dim):
            #print(i)
        out, hidden = self.lstm(x, hidden)
        
        #print(out.shape)
        #hidden = torch.transpose(hidden[0], 1, 0).contiguous()
        #hidden = hidden.view(-1, self.hidden_dim)
        
        out = torch.mean(out, dim=0)
        #print(hidden[0].shape)
        #print(out, hidden[0])
            #print(out.shape)
        #asdf = cwe

        #print(out.shape)
       #print(hidden[0].shape)
        x = self.hid2out(out)
        #print(out.shape)
        #print(hidden[0].shape)
        return x

class LSTMAtt(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, out_dim=1):
        super(LSTMAtt, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=1)

        self.hidden = self.init_hidden()
        
        self.hid2out = nn.Linear(hidden_dim, out_dim)
        self.attn = nn.Linear(self.hidden_dim, 1)
        self.attn_combine = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim //2,
                            num_layers=1, bidirectional=True)
    def init_hidden(self, batch=1):
        return (Variable(torch.randn(2, batch, self.hidden_dim // 2)).cuda(), 
                Variable(torch.randn(2, batch, self.hidden_dim // 2)).cuda())
    

    def forward(self, sentence, test=False):
        batch_size = sentence.shape[0]
        time_dim = sentence.shape[1]
        hidden = self.init_hidden(batch_size)
        
        x = torch.transpose(sentence, 1, 0)
        x = x.unsqueeze(2)
        

        out, hidden = self.lstm(x, hidden)

        out = torch.transpose(out, 1, 0).contiguous()
        out = out.view(time_dim, batch_size, self.hidden_dim)
        
        #print(out.shape)
        #print(hidden[0][0].shape)
        attn_weights = self.attn(out)
        #print(attn_weights.shape)
        attn_weights = F.softmax(attn_weights, dim=0)
        
        
        
        #print(attn_weights)
        #print(attn_weights.shape)
        
        
        attn_weights = attn_weights.permute(1, 2, 0)
        #if not self.training:
        #    print(sentence.data.cpu().numpy())
        #    print(attn_weights.data.cpu().numpy())
        attn_applied = torch.bmm(attn_weights,
                                 out.transpose(1, 0))
        #output = torch.cat((embedded[0], attn_applied[0]), 1)
        #output = self.attn_combine(output).unsqueeze(0)
        output = attn_applied

        output = F.relu(output)
        #print(output.shape)
        #output, hidden = self.gru(output, hidden)

        x = self.hid2out(output)
        #print(out.shape)
        #print(hidden[0].shape)
        if test:
            return x, prob
        return x
    
    def test_att(self, sentence):
  
        return self.forward(sentence, test=True)
    

class IsingCRF(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, spin_dim=16, out_dim=1, bi=True):
        super(IsingCRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.bi = bi
        if bi:
            self.lstm = nn.LSTM(input_dim, hidden_dim //2,
                                num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim,
                                num_layers=1)
        self.hidden = self.init_hidden()
        self.hid2out = nn.Linear(hidden_dim, out_dim)
        self.unary = nn.Linear(hidden_dim, 1)
        self.hid2spin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, spin_dim)
        )
        
        #mask = Variable(torch.eye(hidden_dim)).cuda()
        #self.mask = -1 * (mask - 1)
        #print(self.mask)
        
        self.potential_ = None
        self.unary_ = None
        self.loss = 0
        
    def init_hidden(self, batch=1):
        if self.bi:
            return (Variable(torch.randn(2, batch, self.hidden_dim // 2)).cuda(), 
                    Variable(torch.randn(2, batch, self.hidden_dim // 2)).cuda())
        else:
            return (Variable(torch.randn(1, batch, self.hidden_dim)).cuda(), 
                    Variable(torch.randn(1, batch, self.hidden_dim)).cuda())
    def forward(self, sentence, test=False):
        batch_size = sentence.shape[0]
        time_dim = sentence.shape[1]
        hidden = self.init_hidden(batch_size)
        #print(sentence.shape)
        x = torch.transpose(sentence, 1, 0)
        x = x.unsqueeze(2)
        

        out, hidden = self.lstm(x, hidden)
        #print(out.shape)
        
        
        out = out.transpose(1, 0)
        #print(out.shape)
        spin_m = self.hid2spin(out)
        #print(spin_m.shape)
        
        unary = self.unary(out).squeeze(2)
        #print(unary.shape)
        
        norm = spin_m.norm(dim=2).unsqueeze(2)
        #print(norm.shape)
        #spin_m = spin_m / norm
        potentials_mmm = torch.bmm(spin_m, spin_m.transpose(2, 1))
        
        mask = Variable(torch.eye(time_dim)).cuda()
        mask = -1 * (mask - 1)
        
        potentials_mm = potentials_mmm * mask
        #print(potentials_mm.shape)
        
        prob = torch.sum(potentials_mm, dim=2) / 100 + unary
        #print(prob.shape)
        #print(prob.shape)
        #prob = F.softmax(prob, dim=1)
        #print(prob)
        #print(prob.data.cpu().numpy()[0])
        #print(unary.data.cpu().numpy()[0])
        prob = F.sigmoid(prob)
        #print(prob[0])
        #print(prob.shape)
        
        weighted = torch.bmm(prob.unsqueeze(1), out) / time_dim
        
        #print(weighted.shape)
        x = self.hid2out(weighted)
        #print(out.shape)
        #print(hidden[0].shape)
        if test:
            self.unary_ = unary
            self.potential_ = potentials_mmm
            return x, prob
        if self.training:
            self.loss = self.loss_(prob)
        return x
    
    def test_att(self, sentence):
        return self.forward(sentence, test=True)
    
    def loss_(self, weight):
        #return 0
        #print(weight.size(0))
        #print(weight.data.cpu().numpy())
        weight = torch.mean(weight, dim=0)
        #print(weight.shape)
        loss = - torch.sum(torch.log( 0.5 / weight ) * weight) #/ weight.size(0)
        #print(weight.data.cpu().numpy())
        return loss


class MeanFieldCRF(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, spin_dim=16, out_dim=1, bi=True):
        super(MeanFieldCRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.bi = bi
        if bi:
            self.lstm = nn.LSTM(input_dim, hidden_dim //2,
                                num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim,
                                num_layers=1)
        self.hidden = self.init_hidden()
        self.hid2out = nn.Linear(hidden_dim, out_dim)
        self.unary = nn.Linear(hidden_dim, 1)
        self.hid2bi = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        #mask = Variable(torch.eye(hidden_dim)).cuda()
        #self.mask = -1 * (mask - 1)
        #print(self.mask)
        
        self.potential_ = None
        self.unary_ = None
        self.loss = 0
        
    def init_hidden(self, batch=1):
        if self.bi:
            return (Variable(torch.randn(2, batch, self.hidden_dim // 2)).cuda(), 
                    Variable(torch.randn(2, batch, self.hidden_dim // 2)).cuda())
        else:
            return (Variable(torch.randn(1, batch, self.hidden_dim)).cuda(), 
                    Variable(torch.randn(1, batch, self.hidden_dim)).cuda())
    def forward(self, sentence, test=False):
        batch_size = sentence.shape[0]
        time_dim = sentence.shape[1]
        hidden = self.init_hidden(batch_size)
        #print(sentence.shape)
        x = torch.transpose(sentence, 1, 0)
        x = x.unsqueeze(2)
        

        out, hidden = self.lstm(x, hidden)
        #print(out.shape)
        #print(batch)
        binary = Variable(torch.zeros(batch_size, time_dim, time_dim)).cuda()
        for i in range(time_dim):
            for j in range(time_dim):
                x = self.hid2bi(torch.cat((out[i], out[j]), dim=1)).view(batch_size)
                #print(x)
                #print(binary[:,i,j])

                binary[:,i,j] = x
        
        out = out.transpose(1, 0)
        #print(out.shape)
        
        #print(spin_m.shape)
        
        unary = self.unary(out).squeeze(2)
        #print(unary.shape)
        

        #spin_m = self.hid2spin(out)
        
        #norm = spin_m.norm(dim=2).unsqueeze(2)
        #print(norm.shape)
        #spin_m = spin_m / norm
        #potentials_mmm = torch.bmm(spin_m, spin_m.transpose(2, 1))
        
        mask = Variable(torch.eye(time_dim)).cuda()
        mask = -1 * (mask - 1)
        
        potentials_mm = binary * mask
        #print(potentials_mm.shape)
        
        prob = torch.sum(potentials_mm, dim=2) / 100 + unary

        prob = F.sigmoid(prob)
        #print(prob[0])
        #print(prob.shape)
        
        weighted = torch.bmm(prob.unsqueeze(1), out) / time_dim
        
        #print(weighted.shape)
        x = self.hid2out(weighted)
        #print(out.shape)
        #print(hidden[0].shape)
        if test:
            self.unary_ = unary
            self.potential_ = potentials_mmm
            return x, prob
        if self.training:
            self.loss = self.loss_(prob)
        return x
    
    def test_att(self, sentence):
        return self.forward(sentence, test=True)
    
    def loss_(self, weight):
        #return 0
        #print(weight.size(0))
        #print(weight.data.cpu().numpy())
        weight = torch.mean(weight, dim=0)
        #print(weight.shape)
        loss = - torch.sum(torch.log( 0.5 / weight ) * weight) #/ weight.size(0)
        #print(weight.data.cpu().numpy())
        return loss
