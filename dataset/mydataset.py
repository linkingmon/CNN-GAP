from torch.utils.data.dataset import Dataset
import numpy as np


class myDataset(Dataset):

    def __init__(self, x, y, is_train=True):
        self.x = x
        self.y = y
        self.is_train = is_train

    def __getitem__(self, idx):
        def get_rand():
            return np.random.randint(2)

        def random_flip(x_in, rand):
            if rand == 1:
                return x_in * (-1)
            else:
                return x_in

        def random_scale(x_in, rand):
            if rand == 1:
                return x_in * np.random.uniform(0.5, 2.5)
            else:
                return x_in
        if self.is_train:
            x_t = random_flip(self.x[idx], rand=get_rand())
            x_t = random_scale(x_t, rand=get_rand())
        else:
            x_t = self.x[idx]
        return x_t, self.y[idx]

    def __len__(self):
        return len(self.x)
