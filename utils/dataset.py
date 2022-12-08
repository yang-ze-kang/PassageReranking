from torch.utils.data import Dataset
import torch

class XYDataset(Dataset):
    
    def __init__(self, X, Y) -> None:
        super().__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    @staticmethod
    def collate_fn(batch):
        X, Y = tuple(zip(*batch))
        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)
        return X, Y