#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, r'../')
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import torch.nn as nn
import torch
import pickle
from sklearn.model_selection import train_test_split
import os
from utils.preprocess import pad_sequences
from utils.F1 import F1
from net.FS import FS
from net.FE import FE
from net.net2 import CNN_GAP2
from dataset.mydataset import myDataset
from utils.freeze_model import freeze, unfreeze


# In[2]:


Xtrain = None
with open('../Xtrain', 'rb') as fp:
    Xtrain = pickle.load(fp)

Ytrain = np.load('../Ytrain.npy')

print('# of Xtrain:', len(Xtrain))
print('Shape of Ytrain', Ytrain.shape)


# In[3]:


idx = np.where(Ytrain == 3)
Xtrain_won = np.delete(Xtrain, idx, axis=0)
Ytrain_won = np.delete(Ytrain, idx, axis=0)
print(Xtrain_won.shape)
print(Ytrain_won.shape)


# In[4]:


Xtrain_new = []
Ytrain_new = []
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


# In[5]:


train_data = np.array(Xtrain_new).reshape(len(Ytrain_new), 1, cut_len)
train_label = np.array(Ytrain_new)
print(train_data.shape, train_label.shape)


# In[6]:


train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label, test_size=0.3, random_state=42)
train_data = torch.from_numpy(train_data).float()
valid_data = torch.from_numpy(valid_data).float()
train_label = torch.from_numpy(train_label).long()
valid_label = torch.from_numpy(valid_label).long()
print(train_data.shape, train_label.shape, valid_data.shape, valid_label.shape)


# In[7]:


# create folder
filename = 'model4'
print('All model will ne save in folder: ', filename)
try:
    os.mkdir(filename)
except:
    pass


# In[8]:


write_log = open(filename + '/training.log', 'w+')


# In[9]:


def run(loader, is_train=True):
    acc = 0
    running_loss = 0.0
    f1_score = F1()
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        if is_train:
            optimizer.zero_grad()

        # forward + backward + optimize
        outputs = sel(enc(inputs))
        predict_labels = torch.argmax(outputs, dim=1)
        f1_score.update(predict_labels, labels)
        loss = criterion(outputs, labels)
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


# In[10]:


batch_size = 20
trainloader = DataLoader(myDataset(train_data, train_label), batch_size=batch_size)
validloader = DataLoader(myDataset(valid_data, valid_label, is_train=False), batch_size=batch_size)  # no aug. for valid data

class_weight = compute_class_weight('balanced', [0, 1, 2], train_label.numpy())

enc = FE(final_len=int(cut_len/8))
sel = FS()
enc = enc.cuda()
sel = sel.cuda()

criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight).float()).cuda()
params = list(enc.parameters()) + list(sel.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

highest_f1 = 0
for epoch in range(100):
    # train
    if epoch % 5 == 0:
        sel = unfreeze(sel)
        if epoch == 0:
            enc = enc.train()
            enc = unfreeze(enc)
        else:
            enc = enc.eval()
            enc = freeze(enc)
    else:
        enc = enc.train()
        enc = unfreeze(enc)
        sel = freeze(sel)
    acc, running_loss, train_f1 = run(trainloader)
    train_acc = acc/train_label.size()[0]
    train_loss = running_loss/train_label.size()[0]

    # validate
    enc = enc.eval()
    sel = sel.eval()
    # freeze all model
    enc = freeze(enc)
    sel = freeze(sel)
    
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
        highest_f1 = valid_f1
