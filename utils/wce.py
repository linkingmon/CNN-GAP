import torch 

def wce(predict, target, n_class=3):
    # convert target to one-hot
    z = torch.zeros(target.size()[0], n_class).float()
    z[torch.arange(target.size()[0]),target] = 1
    # calculate f1-loss
    predict = torch.log_softmax(predict, dim=1)
    tp = torch.sum(predict*z, dim=0)
    # p = tp / (predict.sum(dim=0) + 1e-8)
    r = tp / (z.sum(dim=0) + 1e-8)
    # f1 = 2*p*r / (p + r + 1e-8)
    return -r.mean()