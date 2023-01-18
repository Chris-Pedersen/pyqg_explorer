import torch
import math
import numpy as np
import pyqg_explorer.util.transforms as transforms
from torch.utils.data import Dataset


class ForcingDataset(Dataset):
    """
    Subgrid forcing maps dataset
    """
    def __init__(self,x_xarr,y_xarr,seed=42,normalise=True,subsample=None,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        pv:          xarray of the PV field
        dqbar_dt:    xarray of PV tendency
        s:           xarray of the subgrid forcing field
        seed:        random seed used to create train/valid/test splits
        normalise:   bool, normalise mean and variance of fields
        subsample:   None or int: if int, subsample the dataset to a total of N=subsample maps
        train_ratio: proportion of dataset to use as training data
        valid_ratio: proportion of dataset to use as validation data
        test_ratio:  proportion of dataset to use as test data
        
        """
        super().__init__()
        self.normalise=normalise
        self.x_data=torch.unsqueeze(torch.tensor(x_xarr.to_numpy()),dim=1)
        self.y_data=torch.unsqueeze(torch.tensor(y_xarr.to_numpy()),dim=1)
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.rng = np.random.default_rng(seed)

        ## Subsample datasets if required
        self.subsample=subsample
        if self.subsample:
            self.x_data=self.x_data[:self.subsample]
            self.y_data=self.y_data[:self.subsample]

        ## Get means, stds for preprocessing
        self.q_mean=torch.tensor([self.x_data.mean()])
        self.q_std=torch.tensor([self.x_data.std()])
        self.s_mean=torch.tensor([self.y_data.mean()])
        self.s_std=torch.tensor([self.y_data.std()])

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
        if self.normalise:
            return (transforms.normalise_field(self.x_data[idx],self.q_mean,self.q_std),
                        transforms.normalise_field(self.y_data[idx],self.s_mean,self.s_std))
        else:
            return (self.x_data[idx],self.y_data[idx])


## Build a single-step dataset
class SingleStepDataset(Dataset):
    """
    Subgrid forcing maps dataset
    """
    def __init__(self,pv,dqbar_dt,s,seed=42,normalise=True,subsample=None,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        pv:          xarray of the PV field
        dqbar_dt:    xarray of PV tendency
        s:           xarray of the subgrid forcing field
        seed:        random seed used to create train/valid/test splits
        normalise:   bool, normalise mean and variance of fields
        subsample:   None or int: if int, subsample the dataset to a total of N=subsample maps
        train_ratio: proportion of dataset to use as training data
        valid_ratio: proportion of dataset to use as validation data
        test_ratio:  proportion of dataset to use as test data
        
        """
        super().__init__()
        ## Convert xarray data to torch tensor, and reshape
        self.pv=torch.unsqueeze(torch.tensor(pv.to_numpy()),dim=1)
        self.dqbar_dt=torch.unsqueeze(torch.tensor(dqbar_dt.to_numpy()),dim=1)
        self.s=torch.unsqueeze(torch.tensor(s.to_numpy()),dim=1)
        
        ## Split dataset into times i and i+1
        split_idx=np.arange(0,len(pv),2)
        self.y_data=self.pv[split_idx+1]
        self.pv=self.pv[split_idx]
        self.dqbar_dt=self.dqbar_dt[split_idx]
        self.s=self.s[split_idx]
        ## Concat into x_data
        self.x_data=torch.cat((self.pv,self.dqbar_dt,self.s),1)
        
        ## Subsample datasets if required
        self.subsample=subsample
        if self.subsample:
            self.x_data=self.x_data[:self.subsample]
            self.y_data=self.y_data[:self.subsample]
        
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.normalise=normalise
        self.rng = np.random.default_rng(seed)

        self.x_data=self.x_data
        self.y_data=self.y_data
        self.len=len(self.x_data)
        
        assert len(self.x_data)==len(self.y_data), "Number of x and y samples should be the same"
        
        self._get_split_indices()
        
        ## Get means, stds for preprocessing
        means=self.x_data.mean(dim=[0,2,3])
        stds=self.x_data.std(dim=[0,2,3])
        self.q_mean=means[0]
        self.f_mean=means[1]
        self.s_mean=means[2]
        self.q_std=stds[0]
        self.f_std=stds[1]
        self.s_std=stds[2]
        
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
        if self.normalise=="proper":
            ## Return normalised arrays
            pv_out=transforms.normalise_field(self.x_data[idx][0],self.q_mean,self.q_std)
            f_out=transforms.normalise_field(self.x_data[idx][1],self.f_mean,self.f_std)
            s_out=transforms.normalise_field(self.x_data[idx][2],self.s_mean,self.s_std)
            pv_plusone_out=transforms.normalise_field(self.y_data[idx][0],self.q_mean,self.q_std)
            x_out=torch.stack((pv_out,f_out,s_out),dim=0)
            return (x_out,pv_plusone_out)
        elif self.normalise=="euler":
            ## Return normalised arrays
            pv_out=self.x_data[idx][0]/self.q_std
            f_out=(self.x_data[idx][1]*3600)/self.q_std
            s_out=(self.x_data[idx][2]*3600)/self.q_std
            pv_plusone_out=self.y_data[idx]/self.q_std
            x_out=torch.stack((pv_out,f_out,s_out),dim=0)
            return (x_out,pv_plusone_out)
        else:
            ## Return un-normalised arrays
            return (self.x_data[idx],self.y_data[idx])
            