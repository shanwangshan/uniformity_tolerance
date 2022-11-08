import torch
import os
import torch.nn.functional as F
baseline_dir = './t1.0a0.9/'
#baseline_dir = './saved_h_z_amcbest/'

files = os.listdir(baseline_dir)

all_h_data = []
all_z_data = []
all_labels = []
for i in files:
    #__import__("pdb").set_trace()
    data = torch.load(baseline_dir+i,map_location=torch.device('cpu'))



    all_h_data.append(torch.cat((data['h'][0], data['h'][1]),0))
    all_z_data.append(torch.cat((data['z'][0], data['z'][1]),0))
    all_labels.append(torch.cat((data['labels'],data['labels']), 0))


all_h_data =F.normalize( torch.vstack(all_h_data),dim =1)
all_z_data = F.normalize(torch.vstack(all_z_data),dim=1)
all_labels = torch.cat((all_labels))


all_sim = []
for i in range(20):
    indx = torch.where(all_labels==i)
    sim = all_z_data[indx]@(all_z_data[indx].T)
    mask = torch.eye(sim.shape[0], dtype=torch.bool)
    sim = sim[~mask].view(sim.shape[0], -1)
    sim = sim.view(-1)
    all_sim.append(sim)



all_sim = torch.cat((all_sim))
print(torch.mean(all_sim))
