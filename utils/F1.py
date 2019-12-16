import torch


class F1():
    def __init__(self, n_class=3):
        self.num = 0
        self.num_predict = 0
        self.confusion_matrix = torch.tensor([[0]*n_class]*n_class)
        self.n_class = n_class

    def update(self, predicts, groundTruth):
        for i in range(predicts.size()[0]):
            self.confusion_matrix[groundTruth[i]][predicts[i]] += 1

    def get_score(self):
        total_num = torch.sum(self.confusion_matrix, dim=0) + torch.sum(self.confusion_matrix, dim=1)
        f1 = 0
        for i in range(self.n_class):
            f1 += 2*self.confusion_matrix[i][i].float()/total_num[i].float()
        return (f1.item() / self.n_class)
