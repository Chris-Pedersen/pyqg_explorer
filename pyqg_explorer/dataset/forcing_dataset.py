import torch
import math
import numpy as np
from torch.utils.data import Dataset


## Build a single-step dataset
class SingleStepDataset(Dataset):
    """
    Subgrid forcing maps dataset
    """
    def __init__(self,pv,dqbar_dt,s,seed=42,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        pv:          xarray of the PV field
        dqbar_dt:    xarray of PV tendency
        s:           xarray of the subgrid forcing field
        seed:        random seed used to create train/valid/test splits
        train_ratio: proportion of dataset to use as training data
        valid_ratio: proportion of dataset to use as validation data
        test_ratio:  proportion of dataset to use as test data
        
        """
        super().__init__()
        self.pv=torch.unsqueeze(torch.tensor(pv.to_numpy()),dim=1)
        self.dqbar_dt=torch.unsqueeze(torch.tensor(dqbar_dt.to_numpy()),dim=1)
        self.s=torch.unsqueeze(torch.tensor(s.to_numpy()),dim=1)
        ## Generate array for Q_i+1
        self.pv_plusone=torch.roll(self.pv,1,dims=0)
        
        ## Drop last index, where we have no i+1
        self.pv=self.pv[:-1, :, :, :]
        self.dqbar_dt=self.dqbar_dt[:-1, :, :, :]
        self.s=self.s[:-1, :, :, :]
        self.pv_plusone=self.pv_plusone[1:, :, :, :]
        
        ## Cat into x_data
        self.x_data=torch.cat((self.pv,self.dqbar_dt,self.s),1)
        self.y_data=self.pv_plusone
        
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.rng = np.random.default_rng(seed)

        self.x_renorm=torch.std(self.x_data)
        self.y_renorm=torch.std(self.y_data)
        self.x_data=self.x_data/self.x_renorm
        self.y_data=self.y_data/self.y_renorm
        self.len=len(self.x_data)
        
        assert len(self.x_data)==len(self.y_data), "Number of x and y samples should be the same"
        
        self._get_split_indices()
        
    def _get_split_indices(self):
        """ Set indices for train, valid and test splits """

        ## Randomly shuffle indices of entire dataset
        rand_indices=self.rng.permutation(np.arange(self.len))

        ## Set number of train, valid and test points
        num_train=math.floor(self.len*self.train_ratio)
        num_valid=math.floor(self.len*self.valid_ratio)
        num_test=math.floor(self.len*self.test_ratio)
        
        ## Make sure we aren't overcounting
        assert (num_train+num_valid+num_test) <= self.len
        
        ## Pick train, test and valid indices from shuffled list
        self.train_idx=rand_indices[0:num_train]
        self.valid_idx=rand_indices[num_train+1:num_train+num_valid]
        self.test_idx=rand_indices[len(self.valid_idx)+1:]
        
        ## Make sure there's no overlap between train, valid and test data
        assert len(set(self.train_idx) & set(self.valid_idx) & set(self.test_idx))==0, (
                "Common elements in train, valid or test set")
        
        
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.x_data[idx],self.y_data[idx])

class ForcingDataset(Dataset):
    """
    Subgrid forcing maps dataset
    """
    def __init__(self,x_xarr,y_xarr,seed=42,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        x_xarr:      xarray of "x" data, i.e. the low resolution fields
        y_xarr:      xarray of "y" data, i.e. the subgrid forcing field
        seed:        random seed used to create train/valid/test splits
        train_ratio: proportion of dataset to use as training data
        valid_ratio: proportion of dataset to use as validation data
        test_ratio:  proportion of dataset to use as test data
        
        """
        super().__init__()
        self.x_data=torch.unsqueeze(torch.tensor(x_xarr.to_numpy()),dim=1)
        self.y_data=torch.unsqueeze(torch.tensor(y_xarr.to_numpy()),dim=1)
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.rng = np.random.default_rng(seed)

        self.x_renorm=torch.std(self.x_data)
        self.y_renorm=torch.std(self.y_data)
        self.x_data=self.x_data/self.x_renorm
        self.y_data=self.y_data/self.y_renorm
        self.len=len(self.x_data)
        
        assert len(self.x_data)==len(self.y_data), "Number of x and y samples should be the same"
        
        self._get_split_indices()
        
    def _get_split_indices(self):
        """ Set indices for train, valid and test splits """

        ## Randomly shuffle indices of entire dataset
        rand_indices=self.rng.permutation(np.arange(self.len))

        ## Set number of train, valid and test points
        num_train=math.floor(self.len*self.train_ratio)
        num_valid=math.floor(self.len*self.valid_ratio)
        num_test=math.floor(self.len*self.test_ratio)
        
        ## Make sure we aren't overcounting
        assert (num_train+num_valid+num_test) <= self.len
        
        ## Pick train, test and valid indices from shuffled list
        self.train_idx=rand_indices[0:num_train]
        self.valid_idx=rand_indices[num_train+1:num_train+num_valid]
        self.test_idx=rand_indices[len(self.valid_idx)+1:]
        
        ## Make sure there's no overlap between train, valid and test data
        assert len(set(self.train_idx) & set(self.valid_idx) & set(self.test_idx))==0, (
                "Common elements in train, valid or test set")
        
        
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.x_data[idx],self.y_data[idx])