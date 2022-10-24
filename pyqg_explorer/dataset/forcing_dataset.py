import torch
import math
import numpy as np
from torch.utils.data import Dataset


class ForcingDataset(Dataset):
    """
    Subgrid forcing maps dataset
    """
    def __init__(self,x_xarr,y_xarr,seed=42,train_ratio=0.8,valid_ratio=0.1,test_ratio=0.1):
        """
        x_xarr:      xarray of "x" data, i.e. the low resolution fields
        y_xarr:      xarray of "y" data, i.e. the subgrid forcing field
        seed:        random seed used to create train/valid/test splits
        train_ratio: proportion of dataset to use as training data
        valid_ratio: proportion of dataset to use as validation data
        test_ratio:  proportion of dataset to use as test data
        
        """
        super().__init__()
        self.x_data=torch.tensor(x_xarr.to_numpy())
        self.y_data=torch.tensor(y_xarr.to_numpy())
        self.len=len(self.x_data)
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.rng = np.random.default_rng(seed)
        
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