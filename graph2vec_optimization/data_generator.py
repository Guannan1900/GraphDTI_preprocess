from torch.utils import data
import numpy as np


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, path):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.path = path

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        y = self.labels[ID]

        if y == 1:
            folder = self.path + 'positive/'
        elif y == 0:
            folder = self.path + 'negative/'

        # Load data and get label
        X = np.load(folder + ID, allow_pickle=True)
        X_g2v = X[812:1412]

        return X_g2v, y