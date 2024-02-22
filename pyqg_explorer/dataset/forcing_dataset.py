import torch
import math
import json
import numpy as np
import xarray as xr
import pyqg_explorer.util.transforms as transforms
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """ Base object to store core dataset methods """
    def __init__(self,seed=42,subsample=None,drop_spin_up=True,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        super().__init__()
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.subsample=subsample
        self.seed=seed
        self.rng = np.random.default_rng(self.seed)

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
        subsample:     None or int: if int, subsample the dataset to a total of N=subsample maps
        drop_spin_up:  Drop all snapshots taken during the spin-up phase
        train_ratio:   proportion of dataset to use as training data
        valid_ratio:   proportion of dataset to use as validation data
        test_ratio:    proportion of dataset to use as test data
        """
        super().__init__(subsample=subsample,seed=seed)
        
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
    def __init__(self,file_path,seed=42,subsample=None,drop_spin_up=False,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        file_path:   path to data
        seed:        random seed used to create train/valid/test splits
        subsample:   None or int: if int, subsample the dataset to a total of N=subsample maps
        drop_spin_up:  Drop all snapshots taken during the spin-up phase
        train_ratio: proportion of dataset to use as training data
        valid_ratio: proportion of dataset to use as validation data
        test_ratio:  proportion of dataset to use as test data
        
        """
        super().__init__(subsample=subsample,seed=seed)

        self.drop_spin_up=drop_spin_up
        data_full=xr.open_dataset(file_path)
        if self.drop_spin_up:
            data_full=data_full.sel(time=slice(2*100800000.0,5.096036e+08))
        
        def concat_arrays(xarray_subdata):
            def collapse_and_reshape(xarray):
                return torch.tensor(xarray.stack(snapshot=("run","time")).transpose("snapshot","lev","y","x").data)
            channel_index=1
            return torch.cat([collapse_and_reshape(xarray) for xarray in xarray_subdata], channel_index)
        
        if list(data_full.attrs.keys())[0][0:5]=="torch":
            all_data=concat_arrays([data_full.q,data_full.S])
            all_data=torch.tensor(all_data,dtype=torch.float32)
        else:
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
        subsample:     None or int: if int, subsample the dataset to a total of N=subsample maps
        drop_spin_up:  Drop all snapshots taken during the spin-up phase
        train_ratio:   proportion of dataset to use as training data
        valid_ratio:   proportion of dataset to use as validation data
        test_ratio:    proportion of dataset to use as test data
        
        """
        super().__init__(subsample=subsample,seed=seed)
        
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
    def __init__(self,file_path,subgrid_models=["CNN","ZB","BScat","HRC"],channels=4,seed=42,subsample=None,drop_spin_up=False,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        file_path:       path to data
        subgrid_models:  List containing subgrid models: can have any of: ["CNN", "ZB", "BScat", "HRC"]
                         "HRC" stands for a high-res, coarsened system, with the diagnosed forcing that we would use to train
                         an offline model. All other systems are run in low res, with some subgrid model
        channels:        2 or 4 - 2 channels will return only the q field, 4 channels will also return the subgrid forcing
        seed:            random seed used to create train/valid/test splits
        subsample:       None or int: if int, subsample the dataset to a total of N=subsample maps
        drop_spin_up:    Drop all snapshots taken during the spin-up phase
        train_ratio:     proportion of dataset to use as training data
        valid_ratio:     proportion of dataset to use as validation data
        test_ratio:      proportion of dataset to use as test data
        
        """
        super().__init__(subsample=subsample,seed=seed)
        
        self.drop_spin_up=drop_spin_up
        self.subgrid_models=subgrid_models
        self.file_path=file_path
        self.subsample=subsample
        self.channels=channels

        x=[]
        y=[]
        for subgrid_model in self.subgrid_models:
            file_string=self.file_path+"all_"+subgrid_model+".nc"
            all_data=self._build_data(file_string,subgrid_model)
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
        
    def _build_data(self,file_path,subgrid_model):
        """ For a given xarray file, slice, reshape and place the data in a torch tensor. If a subgrid
            forcing field is required, store this too """
        data_full=xr.open_dataset(file_path)
        if self.drop_spin_up:
            data_full=data_full.sel(time=slice(100800000.0,5.096036e+08))
        
        def concat_arrays(xarray_subdata):
            def collapse_and_reshape(xarray):
                return torch.tensor(xarray.stack(snapshot=("run","time")).transpose("snapshot","lev","y","x").data)
            channel_index=1
            return torch.cat([collapse_and_reshape(xarray) for xarray in xarray_subdata], channel_index)

        if self.channels==4 and subgrid_model=="HRC":
            all_data=concat_arrays([data_full.q,data_full.q_forcing_advection])
        elif self.channels==4:
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


class RolloutDataset(BaseDataset):
    """
    x_data is q_i, y_data is s_i
    """
    def __init__(self,increment,rollout,file_path="/scratch/cp3759/pyqg_data/sims/rollouts/",subgrid_models=["CNN","ZB","BScat","HRC"],subgrid_forcing=False,seed=42,subsample=None,drop_spin_up=False,num_sims=274,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        increment:       number of numerical timesteps between snapshots
        rollout:         number of snapshots to store at each datapoint
        file_path:       data directory
        subgrid_models:  List containing subgrid models: can have any of: ["CNN", "ZB", "BScat", "HRC"]
                         "HRC" stands for a high-res, coarsened system, with the diagnosed forcing that we would use to train
                         an offline model. All other systems are run in low res, with some subgrid model
        subgrid_forcing: bool - set True to also store subgrid forcing
        seed:            random seed used to create train/valid/test splits
        subsample:       None or int: if int, subsample the dataset to a total of N=subsample maps
        drop_spin_up:    Drop all snapshots taken during the spin-up phase
        num_sims:        Number of simulations to store (for each subgrid model). Max (and default) is 274
        train_ratio:     proportion of dataset to use as training data
        valid_ratio:     proportion of dataset to use as validation data
        test_ratio:      proportion of dataset to use as test data
        
        """
        super().__init__(subsample=subsample,seed=seed)
        
        self.increment=increment
        self.rollout=rollout
        self.file_path=file_path
        self.subgrid_models=subgrid_models
        self.drop_spin_up=drop_spin_up
        self.subgrid_forcing=subgrid_forcing
        self.subsample=subsample
        self.num_sims=num_sims
        self.cuts=None
        
        self.q_data=torch.tensor([])
        if self.subgrid_forcing:
            self.s_data=torch.tensor([])
        else:
            self.s_data=None
            
        ## Populate dataset
        self._build_dataset()
        
        ## Find normalisation factors
        self.q_mean_upper,self.q_mean_lower=self.q_data.mean(dim=[0,2,3,4])
        self.q_std_upper,self.q_std_lower=self.q_data.std(dim=[0,2,3,4])
        if self.subgrid_forcing:
            self.s_mean_upper,self.s_mean_lower=self.s_data.mean(dim=[0,2,3,4])
            self.s_std_upper,self.s_std_lower=self.s_data.std(dim=[0,2,3,4])
        
        self.len=len(self.q_data)
        ## Generate shuffled list of indices
        self._get_split_indices()
    
    def _get_cuts(self,data):
        """ For a requested increment and rollout, find a list of indices to subsample the correct snapshots from the full dataset """
        data_attrs=json.loads(data.attrs['pyqg_params'])
        self.data_increment=data_attrs["increment"]
        self.data_rollout=data_attrs["rollout"]
        assert self.data_rollout >= self.config["rollout"], "Requested rollout longer than dataset rollout"
        self.num_rollouts=int(len(data.time)/(self.data_rollout+1))
        cuts=np.array([],dtype=int)
        for aa in range(self.num_rollouts):
            cuts=np.append(cuts,aa*(self.data_rollout+1)+np.arange(0,int((self.increment/self.data_increment)*self.rollout+1),int(self.increment/self.data_increment)))
        self.cuts=cuts
        return
    
    def _build_dataset(self):
        """ Loop over subgrid forcing models, and simulation ensembles to populate a dataset. For each sim, subsample from the relevant
            snapshots, and concat these into q_i (and s_i if requested) data """
    
        for subgrid_model in self.subgrid_models:
            if subgrid_model=="HRC":
                subgrid_model="None"
            for aa in range(1,self.num_sims):
                data_path=self.file_path+"rollout_"+subgrid_model+"_"+str(aa)+".nc"
                sim_data=self._load_and_cut_data(data_path,subgrid_model,self.subgrid_forcing)
                self.q_data=torch.cat((self.q_data,sim_data[0]))
                if self.subgrid_forcing:
                    self.s_data=torch.cat((self.s_data,sim_data[1]))
                    
        return
    
    def _load_and_cut_data(self,data_path,subgrid_model,subgrid_forcing=False):
        """ For a given simulation, extract the relevant snapshots. Concat into torch tensors of the appropriate shape. Concat to the self.x
        (and self.y) datasets """

        data_full=xr.open_dataset(data_path)

        ## Make relevant cuts if we are dropping spin-up snapshots
        if self.drop_spin_up==True:
            ## Hardcode the timeslice assuming sampling freq of 1000 timesteps
            data_full=data_full.sel(time=slice(100700000.0,5.096036e+08))
        
        ## If we have not yet found cut indices, find these now
        if self.cuts is None:
            self._get_cuts(data_full)
            
        ## Else, verify that data increments/rollouts are consistent
        data_attrs=json.loads(data_full.attrs['pyqg_params'])
        assert self.data_increment==data_attrs["increment"] and self.data_rollout==data_attrs["rollout"], "Dataset increments/rollouts are inconsistent"
        
        ## Reshape
        torch_q=torch.tensor(data_full.q[self.cuts].to_numpy()).view(self.num_rollouts,self.rollout+1,2,64,64)

        ## Swap axes to have [batch_idx,channels,layer,nx,ny]
        torch_q=torch.swapaxes(torch_q,1,2)

        if subgrid_forcing:
            if subgrid_model=="None":
                torch_s=torch.tensor(data_full.q_forcing_advection[self.cuts].to_numpy()).view(self.num_rollouts,self.rollout+1,2,64,64)
            else:
                torch_s=torch.tensor(data_full.q_subgrid_forcing[self.cuts].to_numpy()).view(self.num_rollouts,self.rollout+1,2,64,64)
            ## Swap axes to have [batch_idx,channels,layer,nx,ny]
            torch_s=torch.swapaxes(torch_s,1,2)
            return [torch_q,torch_s]
        else:
            return [torch_q]
        
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ## Return normalised arrays
        q_upper=transforms.normalise_field(self.q_data[idx,0],self.q_mean_upper,self.q_std_upper)
        q_lower=transforms.normalise_field(self.q_data[idx,1],self.q_mean_lower,self.q_std_lower)
        q_out=torch.stack((q_upper,q_lower),dim=0)
        if self.subgrid_forcing: 
            s_upper=transforms.normalise_field(self.s_data[idx,0],self.s_mean_upper,self.s_std_upper)
            s_lower=transforms.normalise_field(self.s_data[idx,1],self.s_mean_lower,self.s_std_lower)
            s_out=torch.stack((s_upper,s_lower),dim=0)
            return (q_out,s_out)
        else:
            return q_out

class EmulatorDatasetTorch(BaseDataset):
    """
    Load rollout datasets for torchqg sims, for the purposes of training an emulator
    over multiple recurrent passes
    """
    def __init__(self,increment,rollout,file_path="/scratch/cp3759/pyqg_data/sims/torchqg_sims/",eddy=True,seed=42,subsample=None,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        increment:       number of numerical timesteps between snapshots
        rollout:         number of snapshots to store at each datapoint
        file_path:       data directory
        eddy:            flag to decide if we are using eddy or jet sims
        seed:            random seed used to create train/valid/test splits
        subsample:       None or int: if int, subsample the dataset to a total of N=subsample maps
        train_ratio:     proportion of dataset to use as training data
        valid_ratio:     proportion of dataset to use as validation data
        test_ratio:      proportion of dataset to use as test data
        
        """
        super().__init__(subsample=subsample,seed=seed)
        
        self.increment=increment
        self.rollout=rollout
        self.file_path=file_path
        self.subsample=subsample
        self.increment=increment
        self.rollout=rollout
        self.cuts=None
        if eddy==True:
            self.sim_config="eddy"
        else:
            self.sim_config="jet"
        
        file_path=file_path+"%d_step/all_%s.nc" % (self.increment,self.sim_config)
        data_full=xr.open_dataset(file_path)
        self._get_cuts(data_full)
        self.x_data=torch.tensor(data_full.q[:,self.cuts].values,dtype=torch.float32)
        ## We are dealing with a fair whack of memory here, so don't hang around for garbage collection
        del(data_full)

        ## View [sim index, snapshot index, layer index, nx, ny] -> [sim index, trajectory index, rollout index, layer index, nx, ny]
        self.x_data=self.x_data.view(self.x_data.shape[0],int(self.x_data.shape[1]/(self.rollout+1)),int(self.rollout+1),self.x_data.shape[-3],self.x_data.shape[-2],self.x_data.shape[-1])
        ## Reshape [sim index, trajectory index, rollout index, layer index, nx, ny] -> [batch index, rollout index, layer index, nx, ny]
        self.x_data=self.x_data.reshape(self.x_data.shape[0]*self.x_data.shape[1],self.x_data.shape[-4],self.x_data.shape[-3],self.x_data.shape[-2],self.x_data.shape[-1])
        
        ## Find normalisation factors
        self.q_mean_upper,self.q_mean_lower=self.x_data.mean(dim=[0,1,3,4])
        self.q_std_upper,self.q_std_lower=self.x_data.std(dim=[0,1,3,4])
        
        self.len=len(self.x_data)
        ## Subsample datasets if required
        if self.subsample:
            self.x_data=self.x_data[:self.subsample]
            
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.rng = np.random.default_rng(seed)

        self.len=len(self.x_data)
        ## Generate shuffled list of indices
        self._get_split_indices()
    
    def _get_cuts(self,data):
        """ For a requested increment and rollout, find a list of indices to subsample the correct snapshots from the full dataset """
        data_attrs=json.loads(data.attrs['rollout_config'])
        self.data_increment=data_attrs["increment"]
        self.data_rollout=data_attrs["rollout"]
        self.num_rollouts=int(len(data.time)/(self.data_rollout+1))
        cuts=np.array([],dtype=int)
        for aa in range(self.num_rollouts):
            cuts=np.append(cuts,aa*(self.data_rollout+1)+np.arange(0,int((self.increment/self.data_increment)*self.rollout+1),int(self.increment/self.data_increment)))
        self.cuts=cuts
        return

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ## Return normalised arrays
        x_upper=transforms.normalise_field(self.x_data[idx,:,0],self.q_mean_upper,self.q_std_upper)
        x_lower=transforms.normalise_field(self.x_data[idx,:,1],self.q_mean_lower,self.q_std_lower)
        x_out=torch.stack((x_upper,x_lower),dim=1)
        return x_out
