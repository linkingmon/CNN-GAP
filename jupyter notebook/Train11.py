#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, r'../')
import os
from sklearn.model_selection import train_test_split
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from utils.preprocess import pad_sequences
from utils.F1 import F1
from net.FS import FS
from net.FSwn import FSwn
from net.FE import FE
from dataset.mydataset import myDataset
from utils.freeze_model import freeze, unfreeze
from utils.f1_loss import f1_loss


# In[2]:


Xtrain = None
with open('../Xtrain', 'rb') as fp:
    Xtrain = pickle.load(fp)

Ytrain = np.load('../Ytrain.npy')

print('# of Xtrain:', len(Xtrain))
print('Shape of Ytrain', Ytrain.shape)


# In[3]:

Xtrain = np.array(Xtrain)
idx = np.where(Ytrain == 3)[0]
Xtrain_won = np.delete(Xtrain, idx, axis=0)
Ytrain_won = np.delete(Ytrain, idx, axis=0)
print("Xtrain wo noise",Xtrain_won.shape)
print("Ytrain wo noise",Ytrain_won.shape)
# avoid error train test split 
Xtrain_noise = Xtrain[idx]
Ytrain_noise = Ytrain[idx]
print("Xtrain noise:", Xtrain_noise.shape)
print("Ytrain noise", Ytrain_noise.shape)

# In[4]:


Xtrain_new = []
Ytrain_new = []
Xtrain_new_noise = [] # avoid error train test split 
Ytrain_new_noise = [] # avoid error train test split 
cut_len = 9000
thres = 0.65
for i in range(len(Xtrain_won)):
    cut = len(Xtrain_won[i]) // cut_len
    for j in range(1, cut+1):
        Xtrain_new.append(Xtrain_won[i][(j-1)*cut_len:j*cut_len])
        Ytrain_new.append(Ytrain_won[i])        
    if len(Xtrain_won[i]) % cut_len >= int(cut_len*thres):
        x_remain = Xtrain_won[i][cut*cut_len:]
        remainder = cut_len - len(x_remain)
        Xtrain_new.append(np.pad(x_remain, (int(remainder/2), remainder - int(remainder/2)), 'constant', constant_values=0))
        Ytrain_new.append(Ytrain_won[i])
# avoid error train test split 
for i in range(len(Xtrain_noise)):
    cut = len(Xtrain_noise[i]) // cut_len
    for j in range(1, cut+1):
        Xtrain_new_noise.append(Xtrain_noise[i][(j-1)*cut_len:j*cut_len])
        Ytrain_new_noise.append(Ytrain_noise[i])        
    if len(Xtrain_noise[i]) % cut_len >= int(cut_len*thres):
        x_remain = Xtrain_noise[i][cut*cut_len:]
        remainder = cut_len - len(x_remain)
        Xtrain_new_noise.append(np.pad(x_remain, (int(remainder/2), remainder - int(remainder/2)), 'constant', constant_values=0))
        Ytrain_new_noise.append(Ytrain_noise[i])

# In[5]:


train_data = np.array(Xtrain_new).reshape(len(Ytrain_new),1,cut_len)
train_label = np.array(Ytrain_new)
train_data_noise = np.array(Xtrain_new_noise).reshape(len(Ytrain_new_noise),1,cut_len)
train_label_noise = np.array(Ytrain_new_noise)
print(train_data.shape, train_label.shape, train_data_noise.shape, train_label_noise.shape)

# In[6]:

# split noise and not-noise seperately (due to using the pretrained model by splitting on not-nosie data)
train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label, test_size=0.3, random_state=42)
train_data_noise, valid_data_noise, train_label_noise, valid_label_noise = train_test_split(train_data_noise, train_label_noise, test_size=0.3, random_state=42)

train_data = np.concatenate([train_data, train_data_noise], axis=0)
valid_data = np.concatenate([valid_data, valid_data_noise], axis=0)
train_label = np.concatenate([train_label, train_label_noise], axis=0)
valid_label = np.concatenate([valid_label, valid_label_noise], axis=0)

# random shuffle the concatenate data
shuffle_idx1 = np.arange(train_label.shape[0])
shuffle_idx2 = np.arange(valid_label.shape[0])
np.random.shuffle(shuffle_idx1)
np.random.shuffle(shuffle_idx2)
train_data = train_data[shuffle_idx1]
train_label = train_label[shuffle_idx1]
valid_data = valid_data[shuffle_idx2]
valid_label = valid_label[shuffle_idx2]

train_data = torch.from_numpy(train_data).float()
valid_data = torch.from_numpy(valid_data).float()
train_label = torch.from_numpy(train_label).long()
valid_label = torch.from_numpy(valid_label).long()
print(train_data.shape, train_label.shape, valid_data.shape, valid_label.shape)


# In[7]:


# create folder
filename = 'model11'
print('All model will ne save in folder: ', filename)
try:
    os.mkdir(filename)
except:
    pass


# In[8]:


write_log = open(filename + '/training.log', 'w+')


# In[9]:


filename2 = 'model12/'
file_list = os.listdir(filename2)
file_list.sort()
end_name = file_list[-2].split('_')[2]


# In[10]:


enc = FE(final_len=int(cut_len/8),d_rate=0.1)
sel = FS()
print("Load from: ",filename2+'model_enc_'+end_name)
print("Load from: ",filename2+'model_sel_'+end_name)
enc.load_state_dict(torch.load(filename2+'model_enc_'+end_name))
sel.load_state_dict(torch.load(filename2+'model_sel_'+end_name))
enc = enc.cuda()
sel = sel.cuda()

def run(loader, is_train=True):
    acc = 0
    running_loss = 0.0
    f1_score = F1(n_class=4)
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        if is_train:
            optimizer.zero_grad()

        # forward + backward + optimize
        out_inter = enc(inputs)
        outputs1 = sel(out_inter)
        outputs2 = sel2(out_inter)
        outputs = torch.cat([outputs1, outputs2], dim = 1)
        predict_labels = torch.argmax(outputs, dim=1)
        f1_score.update(predict_labels, labels)
        loss = f1_loss(outputs, labels, n_class=4)
        temp = (predict_labels == labels)
        acc += sum(temp).item()
        if is_train:
            loss.backward()
            optimizer.step()

        # print statistics
        running_loss += loss.item()
        if is_train:
            print('[Epoch: %3d] [acc:%.4f] [loss: %.4f] [%5d/%5d]' % (epoch + 1, acc/(batch_size*(i+1)), running_loss/(batch_size*(i+1)), batch_size*(i+1), train_label.size()[0]), end='\r')
    return acc, running_loss, f1_score.get_score()


# In[15]:


batch_size = 20
trainloader = DataLoader(myDataset(train_data, train_label), batch_size=batch_size)
validloader = DataLoader(myDataset(valid_data, valid_label, is_train=False), batch_size=batch_size)  # no aug. for valid data

sel2 = FSwn()
sel2 = sel2.cuda()

params = list(enc.parameters()) + list(sel.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4)

highest_f1 = 0
for epoch in range(200):
    # train
    enc = enc.train()
    sel = sel.train()
    sel2 = sel2.train()
    enc = unfreeze(enc)
    sel = unfreeze(sel)
    sel2 = unfreeze(sel2)

    acc, running_loss, train_f1 = run(trainloader)
    train_acc = acc/train_label.size()[0]
    train_loss = running_loss/train_label.size()[0]

    # validate
    enc = enc.eval()
    sel = sel.eval()
    sel2 = sel2.eval()
    enc = freeze(enc)
    sel = freeze(sel)
    sel2 = freeze(sel2)
    
    acc, running_loss, valid_f1 = run(validloader, is_train=False)
    valid_acc = acc/valid_label.size()[0]
    valid_loss = running_loss/valid_label.size()[0]
    print('>>> [Epoch: %3d] [train_acc:%.4f] [train_loss: %.4f] [train_f1:%.4f] [valid_acc:%.4f] [valid_loss: %.4f] [valid_f1:%.4f]\n'
          % (epoch + 1, train_acc, train_loss, train_f1, valid_acc, valid_loss, valid_f1))

    # write history to file
    # Epoch, train_acc, train_loss, train_f1, valid_acc, valid_loss, valid_f1    
    write_log.write('%3d %.6f %.6f %.6f %.6f %.6f %.6f\n'
       % (epoch + 1, train_acc, train_loss, train_f1, valid_acc, valid_loss, valid_f1))

    # save model
    if highest_f1 < valid_f1:
        torch.save(enc.state_dict(), filename+'/model_enc_%05d-%.5f-%.5f-%.5f-%.5f.h5' % (epoch+1, train_acc, train_f1, valid_acc, valid_f1))
        torch.save(sel.state_dict(), filename+'/model_sel_%05d-%.5f-%.5f-%.5f-%.5f.h5' % (epoch+1, train_acc, train_f1, valid_acc, valid_f1))
        torch.save(sel2.state_dict(), filename+'/model_sel2_%05d-%.5f-%.5f-%.5f-%.5f.h5' % (epoch+1, train_acc, train_f1, valid_acc, valid_f1))
        highest_f1 = valid_f1

