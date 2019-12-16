#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import sys
sys.path.insert(0, r'../')
from utils.preprocess import pad_sequences
from utils.F1 import F1
from net.FS import FS
from net.FSwn import FSwn
from net.FE import FE
from dataset.mydataset import myDataset
from scipy.signal import resample


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


try:
    os.mkdir('confusion matrix')
except:
    pass


# In[8]:


filename = 'model'+str(sys.argv[1])+'/'
file_list = os.listdir(filename)
file_list.sort()
end_name = file_list[-2].split('_')[2]


# In[9]:

have_n = os.path.isfile(filename+'model_sel2_'+end_name)
enc = FE(int(cut_len/8))
sel = FS()
sel2 = FSwn()
print("Load from: ",filename+'model_enc_'+end_name)
print("Load from: ",filename+'model_sel_'+end_name)
enc.load_state_dict(torch.load(filename+'model_enc_'+end_name))
sel.load_state_dict(torch.load(filename+'model_sel_'+end_name))
enc = enc.cuda()
sel = sel.cuda()
enc = enc.eval()
sel = sel.eval()
if have_n:
    print("Load from: ",filename+'model_sel2_'+end_name)
    sel2.load_state_dict(torch.load(filename+'model_sel2_'+end_name))
    sel2 = sel2.cuda()
    sel2 = sel2.eval()
print("HAVE Noise:", have_n)



# In[10]:


batch_size = 1
validloader = DataLoader(myDataset(valid_data, valid_label, is_train=False), batch_size=batch_size)  # no aug. for valid data

pred_label = []

for i, data in enumerate(validloader, 0):
    if i % 5 == 0:
        print('Calculate %4d'% i, end = '\r')
    inputs, labels = data
    inputs = inputs.cuda()
    labels = labels.cuda()

    # forward + backward + optimize
    out_inter = enc(inputs)
    outputs1 = sel(out_inter)
    if have_n:
        outputs2 = sel2(out_inter)
        outputs = torch.cat([outputs1, outputs2], dim = 1)
    else:
        outputs = outputs1
    predict_labels = torch.argmax(outputs, dim=1).cpu().numpy()
    pred_label.append(predict_labels)


# In[11]:


from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)
y_test = valid_label.numpy()
print(np.array(pred_label).shape)
y_pred = np.array(pred_label).reshape(-1)
print(y_pred.shape)
class_names = np.array(['N','A','O', '~'])
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('confusion matrix/'+filename[:-1]+'_unnorm.jpg')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('confusion matrix/'+filename[:-1]+'_norm.jpg')

