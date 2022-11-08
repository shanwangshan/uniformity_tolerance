import torch
import os
import numpy as np
#baseline_dir = './saved_h_z_baseline/'
baseline_dir = './t1.0a0.9/'
#baseline_dir = './saved_h_z_amcbest/'
files = os.listdir(baseline_dir)

def uniform_loss(x, t=2):
    x /= x.norm(p=2,dim=1,keepdim=True)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    #return x.pow(2).mul(-t).exp().mean().log()

x = []
y = []
for i in files:
    #__import__("pdb").set_trace()
    #data = torch.load(baseline_dir+i)
    data = torch.load(baseline_dir+i,map_location=torch.device('cpu'))
    #breakpoint()
    #x.append(torch.norm(data['z'][0]-data['z'][1],p=2, dim=1))
    x.append(data['z'][0])
    y.append(data['z'][1])

x = torch.cat(x, dim=0)
y = torch.cat(y, dim=0)
#breakpoint()
k1 = np.random.choice(x.shape[0], x.shape[0], replace=False)
k2 = np.random.choice(y.shape[0], y.shape[0], replace=False)
#__import__("pdb").set_trace()

uloss_x = uniform_loss(x[k1])

#uloss_y = uniform_loss(y)
#u_loss = (uloss_x + uloss_y)/2
print(uloss_x)
