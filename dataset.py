import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random

class Mujoco(Dataset):
    def __init__(self,seq_len,path):
        super().__init__()
        data = np.load(path)
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        start = random.randint(0,len(self.data)-self.seq_len)
        data = self.data[index]
        data = data[start:start+self.seq_len-1,:]
        return data

class Ptbxl(Dataset):
    def __init__(self,seq_len,path):
        super().__init__()
        data = np.load(path)
        self.data = data
        self.seq_len = seq_len
        self.max_seq_len = data.shape[2]
        pass
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        if self.max_seq_len != self.seq_len:
            start = random.randint(0,self.max_seq_len-self.seq_len)
        else:
            start = 0
        data = self.data[index]
        data = data[:,start:start+self.seq_len]
        data = data.T
        return data
    

    
class ettm1(Dataset):
    def __init__(self,seq_len,path):
        super().__init__()
        data = np.load(path)
        self.data = data
        self.seq_len = seq_len
        self.max_seq_len = data.shape[1]
        pass
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        if self.seq_len != self.max_seq_len:
            start = random.randint(0,self.max_seq_len-self.seq_len)
        else:
            start = 0
        data = self.data[index]
        data = data[start:start+self.seq_len,:]
        return data

#class electricity(Dataset):
#    def __init__(self,seq_len,path):
#        super().__init__()
#        data = np.load(path)
#        _, K, L, _ = data.shape
#        data = data.reshape(K,L,-1)
#        self.data = data
#        self.seq_len = seq_len
#        self.max_seq_len = L
#    def __len__(self):
#        return len(self.data)
#    def __getitem__(self,index):
#        start = random.randint(0,self.max_seq_len-self.seq_len)
#        data = self.data[index]
#        data = data[:,start:start+self.seq_len]
#        return data

class electricity(Dataset):
    def __init__(self, seq_len, path, batch_features=37):
        super().__init__()
        data = np.load(path)
        _, K, L, _ = data.shape
        data = data.reshape(K, L, -1)
        self.data = data
        self.seq_len = seq_len
        self.max_seq_len = L
        self.batch_features = batch_features

    def __len__(self):
        return len(self.data) * (self.data.shape[2] // self.batch_features)

    def __getitem__(self, index):
        batch_idx = index // (self.data.shape[2] // self.batch_features)
        feature_idx = (index % (self.data.shape[2] // self.batch_features)) * self.batch_features
        data = self.data[batch_idx, :, feature_idx:feature_idx + self.batch_features]
        start = random.randint(0, self.max_seq_len - self.seq_len)
        data = data[:, start:start + self.seq_len]
        return data

class Solar(Dataset):
    def __init__(self,seq_len,path):
        super().__init__()
        data = np.load(path)
        num_batches, num_samples, num_time_steps, num_features = data.shape
        data = data.reshape(num_batches * num_samples, num_time_steps, num_features)
        self.data = data
        self.seq_len = seq_len
        self.max_seq_len = data.shape[1]
        pass
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        if self.max_seq_len != self.seq_len:
            start = random.randint(0,self.max_seq_len-self.seq_len)
        else:
            start = 0
        data = self.data[index]
        data = data[start:start+self.seq_len,:]
        return data
    
class Physio(Dataset):
    def __init__(self, seq_len, path):
        super().__init__()
        self.data = np.load(path)  # 假设数据维度为 (2800, 48, 35)
        self.seq_len = seq_len
        self.max_seq_len = self.data.shape[1]  # 时间序列的长度 48
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.seq_len != self.max_seq_len:
            start = random.randint(0, self.max_seq_len - self.seq_len)
        else:
            start = 0
        data = self.data[index]
        data = data[start:start + self.seq_len, :]
        return data