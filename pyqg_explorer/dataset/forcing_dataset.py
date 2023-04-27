import torch
import math
import numpy as np
import xarray as xr
import pyqg_explorer.util.transforms as transforms
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self,seed=42,subsample=None,drop_spin_up=True,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        super().__init__()
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.subsample=subsample
        self.rng = np.random.default_rng(seed)

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
        return

    def _subsample(self):
        """ Take a subsample of """
        self.x_data=self.x_data[:self.subsample]
        self.y_data=self.y_data[:self.subsample]
        return


class ResidualDataset(BaseDataset):
    """
    x_data is pyqg field at q_i, y_data is q_{i+dt}-q_i
    """
    def __init__(self,file_path,seed=42,subsample=None,drop_spin_up=False,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        file_path:     path to data
        seed:          random seed used to create train/valid/test splits
        normalise:     "proper" to normalise fields to zero mean and unit variance
        subsample:     None or int: if int, subsample the dataset to a total of N=subsample maps
        drop_spin_up:  Drop all snapshots taken during the spin-up phase
        train_ratio:   proportion of dataset to use as training data
        valid_ratio:   proportion of dataset to use as validation data
        test_ratio:    proportion of dataset to use as test data
        
        """
        super().__init__()
        
        self.drop_spin_up=drop_spin_up
        data_full=xr.open_dataset(file_path)
        if self.drop_spin_up:
            data_full=data_full.sel(time=slice(100800000.0,5.096036e+08))
        
        def concat_arrays(xarray_subdata):
            def collapse_and_reshape(xarray):
                return torch.tensor(xarray.stack(snapshot=("run","time")).transpose("snapshot","lev","y","x").data)
            channel_index=1
            return torch.cat([collapse_and_reshape(xarray) for xarray in xarray_subdata], channel_index)
        
        all_data=concat_arrays([data_full.q])
        split_idx=np.arange(0,len(all_data),2)
        self.x_data=all_data[split_idx]
        self.y_data=all_data[split_idx+1]-all_data[split_idx]
        
        
        ## Subsample datasets if required
        if self.subsample:
            self._subsample()
        
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio

        self.len=len(self.x_data)
        
        self._get_split_indices()
        
        ## Get means, stds for preprocessing
        self.q_mean_upper,self.q_mean_lower=self.x_data.mean(dim=[0,2,3])
        self.q_std_upper,self.q_std_lower=self.x_data.std(dim=[0,2,3])
        self.res_mean_upper,self.res_mean_lower=self.y_data.mean(dim=[0,2,3])
        self.res_std_upper,self.res_std_lower=self.y_data.std(dim=[0,2,3])
        
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ## Return normalised arrays
        q_upper=transforms.normalise_field(self.x_data[idx][0],self.q_mean_upper,self.q_std_upper)
        q_lower=transforms.normalise_field(self.x_data[idx][1],self.q_mean_lower,self.q_std_lower)
        res_upper=transforms.normalise_field(self.y_data[idx][0],self.res_mean_upper,self.res_std_upper)
        res_lower=transforms.normalise_field(self.y_data[idx][1],self.res_mean_lower,self.res_std_lower)
        x_out=torch.stack((q_upper,q_lower),dim=0)
        y_out=torch.stack((res_upper,res_lower),dim=0)
        return (x_out,y_out)


class OfflineDataset(BaseDataset):
    """
    Dataset to prepare q, f and s fields for some given time horizon
    """
    def __init__(self,file_path,seed=42,subsample=None,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        file_path:   path to data
        seed:        random seed used to create train/valid/test splits
        normalise:   bool, normalise mean and variance of fields
        subsample:   None or int: if int, subsample the dataset to a total of N=subsample maps
        train_ratio: proportion of dataset to use as training data
        valid_ratio: proportion of dataset to use as validation data
        test_ratio:  proportion of dataset to use as test data
        
        """
        super().__init__()
        
        data_full=xr.open_dataset(file_path)
        
        def concat_arrays(xarray_subdata):
            def collapse_and_reshape(xarray):
                return torch.tensor(xarray.stack(snapshot=("run","time")).transpose("snapshot","lev","y","x").data)
            channel_index=1
            return torch.cat([collapse_and_reshape(xarray) for xarray in xarray_subdata], channel_index)
        
        all_data=concat_arrays([data_full.q,data_full.q_forcing_advection])
        self.x_data=all_data[:,0:2]
        self.y_data=all_data[:,2:4]
        
        
        ## Subsample datasets if required
        if self.subsample:
            self._subsample()
        
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.rng = np.random.default_rng(seed)

        self.len=len(self.x_data)
        
        assert len(self.x_data)==len(self.y_data), "Number of x and y samples should be the same"
        
        self._get_split_indices()
        
        ## Get means, stds for preprocessing
        self.q_mean_upper,self.q_mean_lower=self.x_data.mean(dim=[0,2,3])
        self.q_std_upper,self.q_std_lower=self.x_data.std(dim=[0,2,3])
        self.s_mean_upper,self.s_mean_lower=self.y_data.mean(dim=[0,2,3])
        self.s_std_upper,self.s_std_lower=self.y_data.std(dim=[0,2,3])

        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ## Return normalised arrays
        q_upper=transforms.normalise_field(self.x_data[idx][0],self.q_mean_upper,self.q_std_upper)
        q_lower=transforms.normalise_field(self.x_data[idx][1],self.q_mean_lower,self.q_std_lower)
        s_upper=transforms.normalise_field(self.y_data[idx][0],self.s_mean_upper,self.s_std_upper)
        s_lower=transforms.normalise_field(self.y_data[idx][1],self.s_mean_lower,self.s_std_lower)
        x_out=torch.stack((q_upper,q_lower),dim=0)
        y_out=torch.stack((s_upper,s_lower),dim=0)
        return (x_out,y_out)


class EmulatorDataset(BaseDataset):
    """
    x_data is q_i, y_data is q_{i+dt}
    """
    def __init__(self,file_path,seed=42,subsample=None,drop_spin_up=False,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        file_path:     path to data
        seed:          random seed used to create train/valid/test splits
        normalise:     "proper" to normalise fields to zero mean and unit variance
        subsample:     None or int: if int, subsample the dataset to a total of N=subsample maps
        drop_spin_up:  Drop all snapshots taken during the spin-up phase
        train_ratio:   proportion of dataset to use as training data
        valid_ratio:   proportion of dataset to use as validation data
        test_ratio:    proportion of dataset to use as test data
        
        """
        super().__init__()
        
        self.drop_spin_up=drop_spin_up
        data_full=xr.open_dataset(file_path)
        if self.drop_spin_up:
            data_full=data_full.sel(time=slice(100800000.0,5.096036e+08))
        
        def concat_arrays(xarray_subdata):
            def collapse_and_reshape(xarray):
                return torch.tensor(xarray.stack(snapshot=("run","time")).transpose("snapshot","lev","y","x").data)
            channel_index=1
            return torch.cat([collapse_and_reshape(xarray) for xarray in xarray_subdata], channel_index)
        
        all_data=concat_arrays([data_full.q])
        split_idx=np.arange(0,len(all_data),2)
        self.x_data=all_data[split_idx]
        self.y_data=all_data[split_idx+1]
        
        ## Subsample datasets if required
        if self.subsample:
            self._subsample()
        
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.rng = np.random.default_rng(seed)

        self.len=len(self.x_data)
        
        assert len(self.x_data)==len(self.y_data), "Number of x and y samples should be the same"
        
        self._get_split_indices()
        
        ## Get means, stds for preprocessing
        self.q_mean_upper,self.q_mean_lower=self.x_data.mean(dim=[0,2,3])
        self.q_std_upper,self.q_std_lower=self.x_data.std(dim=[0,2,3])
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ## Return normalised arrays
        q_upper=transforms.normalise_field(self.x_data[idx][0],self.q_mean_upper,self.q_std_upper)
        q_lower=transforms.normalise_field(self.x_data[idx][1],self.q_mean_lower,self.q_std_lower)
        q_t_upper=transforms.normalise_field(self.y_data[idx][0],self.q_mean_upper,self.q_std_upper)
        q_t_lower=transforms.normalise_field(self.y_data[idx][1],self.q_mean_lower,self.q_std_lower)
        x_out=torch.stack((q_upper,q_lower),dim=0)
        y_out=torch.stack((q_t_upper,q_t_lower),dim=0)
        return (x_out,y_out)


class EmulatorForcingDataset(BaseDataset):
    """
    x_data is q_i, y_data is q_{i+dt}
    """
    def __init__(self,file_path,subgrid_models=["CNN","ZB","BScat"],channels=4,seed=42,subsample=None,drop_spin_up=False,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        file_path:       path to data
        subgrid_models:  List containing subgrid models: can have any of: ["CNN", "ZB", "BScat"]
        seed:            random seed used to create train/valid/test splits
        normalise:       "proper" to normalise fields to zero mean and unit variance
        subsample:       None or int: if int, subsample the dataset to a total of N=subsample maps
        drop_spin_up:    Drop all snapshots taken during the spin-up phase
        train_ratio:     proportion of dataset to use as training data
        valid_ratio:     proportion of dataset to use as validation data
        test_ratio:      proportion of dataset to use as test data
        
        """
        super().__init__()
        
        self.drop_spin_up=drop_spin_up
        self.subgrid_models=subgrid_models
        self.file_path=file_path
        self.subsample=subsample
        self.channels=channels
        
        x=[]
        y=[]
        for subgrid_model in self.subgrid_models:
            file_string=self.file_path+"all_"+subgrid_model+".nc"
            all_data=self._build_data(file_string)
            split_idx=np.arange(0,len(all_data),2)
            x.append(all_data[split_idx])
            y.append(all_data[split_idx+1,0:2,:,:])
            
        ## Make sure all elements in list are same length
        subs = iter(x)
        self.maps_per_model = len(next(subs))
        assert all(len(sub) == self.maps_per_model for sub in subs)
        
        self.x_data=torch.vstack(x)
        self.y_data=torch.vstack(y)
        
        self.len=len(self.x_data)
        
        assert len(self.x_data)==len(self.y_data), "Number of x and y samples should be the same"
        
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.rng = np.random.default_rng(seed)
        
        self._get_split_indices()
        
        ## Get means, stds for preprocessing
        if self.channels==4:
            self.q_mean_upper,self.q_mean_lower,self.s_mean_upper,self.s_mean_lower=self.x_data.mean(dim=[0,2,3])
            self.q_std_upper,self.q_std_lower,self.s_std_upper,self.s_std_lower=self.x_data.std(dim=[0,2,3])
        elif self.channels==2:
            self.q_mean_upper,self.q_mean_lower=self.x_data.mean(dim=[0,2,3])
            self.q_std_upper,self.q_std_lower=self.x_data.std(dim=[0,2,3])
        
    def _build_data(self,file_path):
        data_full=xr.open_dataset(file_path)
        if self.drop_spin_up:
            data_full=data_full.sel(time=slice(100800000.0,5.096036e+08))
        
        def concat_arrays(xarray_subdata):
            def collapse_and_reshape(xarray):
                return torch.tensor(xarray.stack(snapshot=("run","time")).transpose("snapshot","lev","y","x").data)
            channel_index=1
            return torch.cat([collapse_and_reshape(xarray) for xarray in xarray_subdata], channel_index)

        if self.channels==4:
            all_data=concat_arrays([data_full.q,data_full.q_subgrid_forcing])
        elif self.channels==2:
            all_data=concat_arrays([data_full.q])
        else:
            print("Unknown number of channels requested")
        if self.subsample:
            all_data=all_data[:int(self.subsample*2)]
        return all_data
    
    def _get_split_indices(self):
        """ Override the inherited method with a few additional features:
            1. Ensure we have balanced samples of each subgrid forcing model
            2. Save random indices for each model, so that we can test on them independently """

        ## Numerb of train, valid, test indices per data subset
        num_train=math.floor(self.maps_per_model*self.train_ratio)
        num_valid=math.floor(self.maps_per_model*self.valid_ratio)
        num_test=math.floor(self.maps_per_model*self.test_ratio)
        
        ## Temp lists to store indices
        all_train=[]
        all_valid=[]
        all_test=[]

        self.model_splits={}

        for aa, elem in enumerate(self.subgrid_models):
            self.model_splits[elem]={}

            ## Generate list of indices for each subgrid model
            rand_indices=self.rng.permutation(np.arange(aa*self.maps_per_model,(aa+1)*self.maps_per_model))

            ## Incorporate subset into full set
            all_train.append(rand_indices[0:num_train])
            all_valid.append(rand_indices[num_train+1:num_train+num_valid])
            all_test.append(rand_indices[num_train+num_valid+1:])

            ## Also store subset separately for testing
            self.model_splits[elem]["train"]=rand_indices[0:num_train]
            self.model_splits[elem]["valid"]=rand_indices[num_train+1:num_train+num_valid]
            self.model_splits[elem]["test"]=rand_indices[num_train+num_valid+1:]
            
        ## Flatten lists into train, valid, and test splits
        self.train_idx=np.array(all_train).flatten()
        self.valid_idx=np.array(all_valid).flatten()
        self.test_idx=np.array(all_test).flatten()
        
        ## Make sure we aren't overcounting
        assert (len(self.train_idx)+len(self.valid_idx)+len(self.test_idx)) <= self.len
        
        ## Make sure there's no overlap between train, valid and test data
        assert len(set(self.train_idx) & set(self.valid_idx) & set(self.test_idx))==0, (
                "Common elements in train, valid or test set")
        return
        
    def _subsample(self):
        """ Override default subsampling due to more complex dataset """
        raise NotImplementedError
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.channels==4:
            ## Return normalised arrays
            q_upper=transforms.normalise_field(self.x_data[idx][0],self.q_mean_upper,self.q_std_upper)
            q_lower=transforms.normalise_field(self.x_data[idx][1],self.q_mean_lower,self.q_std_lower)
            s_upper=transforms.normalise_field(self.x_data[idx][2],self.s_mean_upper,self.s_std_upper)
            s_lower=transforms.normalise_field(self.x_data[idx][3],self.s_mean_lower,self.s_std_lower)
            q_t_upper=transforms.normalise_field(self.y_data[idx][0],self.q_mean_upper,self.q_std_upper)
            q_t_lower=transforms.normalise_field(self.y_data[idx][1],self.q_mean_lower,self.q_std_lower)
            x_out=torch.stack((q_upper,q_lower,s_upper,s_lower),dim=0)
            y_out=torch.stack((q_t_upper,q_t_lower),dim=0)
            return (x_out,y_out)
        elif self.channels==2:
            ## Return normalised arrays
            q_upper=transforms.normalise_field(self.x_data[idx][0],self.q_mean_upper,self.q_std_upper)
            q_lower=transforms.normalise_field(self.x_data[idx][1],self.q_mean_lower,self.q_std_lower)
            q_t_upper=transforms.normalise_field(self.y_data[idx][0],self.q_mean_upper,self.q_std_upper)
            q_t_lower=transforms.normalise_field(self.y_data[idx][1],self.q_mean_lower,self.q_std_lower)
            x_out=torch.stack((q_upper,q_lower),dim=0)
            y_out=torch.stack((q_t_upper,q_t_lower),dim=0)
            return (x_out,y_out)
