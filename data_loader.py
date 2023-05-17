import glob
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from arabert import ArabertPreprocessor

processor = ArabertPreprocessor(model_name='')
def preprocess(text):
  return processor.preprocess(text)

def unprep(text):
  return processor.unpreprocess(text)

class CODAMADARADataset(Dataset):
    """CODA MADARA Dataset"""

    def __init__(self, path, mode='train'):
        self.dataset_path = os.path.join(path, mode)
        if not os.path.exists(f'{self.dataset_path}.csv'):
            self.files = sorted(glob.glob(f'{self.dataset_path}/*.*'))
            self.df = pd.concat([pd.read_csv(f, sep='\t') for f in self.files],
                                ignore_index=True)[['raw', 'CODA']]
            self.df.to_csv(f'{self.dataset_path}.csv', index=False)
        else:
            self.df = pd.read_csv(f'{self.dataset_path}.csv')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        x = preprocess(row['raw'])
        y = preprocess(row['CODA'])
        return x, y


def get_loaders(path="coda-corpus", batch_size=8, shuffle=True):
    train_dataset = CODAMADARADataset(path, 'train')
    test_dataset = CODAMADARADataset(path, 'test')

    train_size = int(len(train_dataset) * 0.8)
    splits = torch.utils.data.random_split(train_dataset,
                                           [train_size, len(train_dataset)-train_size])

    train_loader = torch.utils.data.DataLoader(splits[0], batch_size=batch_size, shuffle=shuffle)
    val_loader = torch.utils.data.DataLoader(splits[1], batch_size=batch_size, shuffle=shuffle)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, val_loader, test_loader

