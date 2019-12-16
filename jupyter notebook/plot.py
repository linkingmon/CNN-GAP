import numpy as np
import matplotlib.pyplot as plt

def plot_log(folder, line):
    L = open(folder+'/training.log').readlines()
    dat = []
    for j in range(len(L)):
        S = [float(t) for t in L[j][4:].strip('\n').split(' ')]
        dat.append(S)
    dat = np.array(dat)
    for i in idx:
        Z = np.zeros((plot_len))*np.nan
        Z[:min(dat.shape[0],plot_len)] = dat[:min(dat.shape[0],plot_len), i]
        if i < 3:
            plt.plot(np.arange(1,plot_len+1), Z[:plot_len], line+'-')
        else:
            plt.plot(np.arange(1,plot_len+1), Z[:plot_len], line+'-.')

mapping = ['train_acc', 'train_loss', 'train_f1', 'valid_acc', 'valid_loss', 'valid_f1']
color = ['b', 'g', 'r', 'c', 'm', 'k']
idx = [0, 3]  # acc
# idx = [1, 4]  # loss
# idx = [2, 5]  # f1
plot_len = 80

plot_nums = {3,9,12}
leg = []

plot_len = 140
cnt = 0
for plot_num in plot_nums:
    eval('plot_log(\'model' + str(plot_num) +'/\', color[' + str(cnt) + '])')
    leg.append(str(plot_num))
    leg.append(str(plot_num)+'V')
    cnt = cnt + 1

plt.xlabel('epoch')
plt.ylabel(mapping[idx[0]].split('_')[1])
plt.legend(leg)
plt.show()