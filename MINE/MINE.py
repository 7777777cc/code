import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

class Mine(nn.Module):
    def __init__(self, input_size=84*84, hidden_size=100):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.fc1.weight,std=0.02)
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.normal_(self.fc2.weight,std=0.02)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.normal_(self.fc3.weight,std=0.02)
            nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, input):
            output = F.elu(self.fc1(input))
            output = F.elu(self.fc2(output))
            output = self.fc3(output)
            
            return output


def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        
    return mi_lb, t, et

def learn_mine(batch, mine_net, mine_net_optim,  ma_et, ma_rate=0.01):
    joint , marginal = batch
    joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
    marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
    mi_lb , t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)


    loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))

    mine_net_optim.zero_grad()
    autograd.backward(loss)
    mine_net_optim.step()
    
    return mi_lb, ma_et
    
def train_obs(data, mine_net,mine_net_optim, batch_size=100, log_freq=int(1e3)):
    result = list()
    ma_et = 1.
    for idx in range(20):
        for i in range(len(data)):
            batch = data[i]['tro'].squeeze(0).reshape(4,84*84), data[i]['tar'].squeeze(0).reshape(4,84*84)
            mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
            result.append(mi_lb.detach().cpu().numpy())
            if (i+1)%(log_freq)==0:
                print(result[-1])
    
    return result
    
def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]


    # "# load obs data\n",
    # data_list=torch.load('mine_data.pt')


    # mine_net_indep = Mine().cuda()
    # mine_net_optim_indep = optim.Adam(mine_net_indep.parameters(), lr=1e-5)
    # result_indep = train_obs(data_list,mine_net_indep,mine_net_optim_indep)



    # result_indep_ma = ma(result_indep)
    # print(result_indep_ma[-1])
    # plt.plot(range(len(result_indep_ma)),result_indep_ma)
