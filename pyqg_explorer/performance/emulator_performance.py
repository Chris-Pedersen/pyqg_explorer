import numpy as np
import torch
import torch.nn as nn
import pyqg_explorer.util.transforms as transforms
import matplotlib.pyplot as plt
import xarray as xr


class EmulatorPerformance():
    """ Object to store performance tests relevant to neural emulators """
    def __init__(self,network):
        """ network:  Torch model we want to test. Assuming this is a model for the subgrid forcing
            valid_loader: validation loader from EmulatorDatasetTorch """
        
        
        if torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"
        
        self.network=network.to(self.device)
        """ Are we using eddy or jet """
        if network.config["eddy"]:
            self.eddy="eddy"
        else:
            self.eddy="jet"
        
    def _get_next_step(self,q_i):
        """ For a given field at time i, use the attributed model to push the system forward
            to time i+dt (dt is stored as the time horizon in model config) """

        q_i_dt=self.network(q_i.unsqueeze(0))
        q_i_dt=q_i_dt.squeeze()

        return q_i_dt+q_i
    
    def get_short_MSEs(self,return_data=False):
        ds=xr.load_dataset("/scratch/cp3759/pyqg_data/sims/emulator_trajectory_sims/torch_%s_1k.nc" % self.eddy)
        times=np.arange(self.network.config["increment"],ds.q.shape[1],self.network.config["increment"])

        criterion=nn.MSELoss()
        fig=plt.figure()
        ## This can all be parallelised to make use of GPU..
        for aa in range(20):
            mses=np.empty(len(times))
            mses_0=np.empty(len(times))
            q_i=self.normalise(torch.tensor(ds.q[aa,0].to_numpy(),device=self.device,dtype=torch.float32))
            ## Index counter for loss arrays
            cc=0
            init=self.normalise(torch.tensor(ds.q[aa,0].values,device=self.device))
            for bb in range(self.network.config["increment"],ds.q.shape[1],self.network.config["increment"]):
                q_i=self._get_next_step(q_i)
                q_i_true=self.normalise(torch.tensor(ds.q[aa,bb].values,device=self.device))
                mses[cc]=(criterion(q_i_true,q_i))
                mses_0[cc]=(criterion(q_i_true,init))
                cc+=1
                
            plt.title("MSE(truth,emulator) in blue, MSE(truth at t=0, truth at t=i) in red")
            plt.plot(times,mses,color="blue",label="Emulator MSE wrt truth",alpha=0.2)
            plt.plot(times,mses_0,color="red",label="True MSE wrt t=0",alpha=0.2)
            plt.yscale("log")
            plt.ylim(1e-3,5e0)
            plt.xlabel("timestep")
            plt.ylabel("MSE")

        return fig

    def normalise(self,q):
        ## Map from physical to normalised space using the factors used to train the network
        ## Normalise each field individually, then cat arrays back to shape appropriate for a torch model
        x_upper = transforms.normalise_field(q[0],self.network.config["q_mean_upper"],self.network.config["q_std_upper"])
        x_lower = transforms.normalise_field(q[1],self.network.config["q_mean_lower"],self.network.config["q_std_lower"])
        x = torch.stack((x_upper,x_lower),dim=0)
        return x

class DenoiserPerformance():
    """ Object to store performance tests relevant to neural emulators """
    def __init__(self,network,denoiser,denoise_timestep=10,denoise_delay=500,denoise_interval=5):
        """ network:  Torch model we want to test. Assuming this is a model for the subgrid forcing
            valid_loader: validation loader from EmulatorDatasetTorch """
        
        
        if torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"
        
        self.network=network.to(self.device)
        self.denoiser=denoiser
        self.denoise_timestep=denoise_timestep
        self.denoise_delay=denoise_delay
        self.denoise_interval=denoise_interval
        self.denoiser.to(self.device)
        """ Are we using eddy or jet """
        if network.config["eddy"]:
            self.eddy="eddy"
        else:
            self.eddy="jet"
        
    def _get_next_step(self,q_i):
        """ For a given field at time i, use the attributed model to push the system forward
            to time i+dt (dt is stored as the time horizon in model config) """

        q_i_dt=self.network(q_i.unsqueeze(0))
        q_i_dt=q_i_dt.squeeze()

        return q_i_dt+q_i
    
    def get_short_MSEs(self,return_data=False):
        ds=xr.load_dataset("/scratch/cp3759/pyqg_data/sims/emulator_trajectory_sims/torch_%s_1k.nc" % self.eddy)
        times=np.arange(self.network.config["increment"],ds.q.shape[1],self.network.config["increment"])

        criterion=nn.MSELoss()
        fig=plt.figure()
        ## This can all be parallelised to make use of GPU..
        for aa in range(20):
            mses=np.empty(len(times))
            mses_0=np.empty(len(times))
            mses_dn=np.empty(len(times))
            q_i=self.normalise(torch.tensor(ds.q[aa,0].to_numpy(),device=self.device,dtype=torch.float32))
            q_i_dn=q_i
            ## Index counter for loss arrays
            cc=0
            init=self.normalise(torch.tensor(ds.q[aa,0].values,device=self.device))
            
            ## Bool to keep track of when to apply denoiser
            should_denoise=False
            for bb in range(self.network.config["increment"],ds.q.shape[1],self.network.config["increment"]):
                
                
                
                q_i_true=self.normalise(torch.tensor(ds.q[aa,bb].values,device=self.device))
                ## Emulator rollout
                q_i=self._get_next_step(q_i)
                ## Denoised emulator rollout
                q_i_dn=self._get_next_step(q_i_dn)
                q_i_dn=self.denoiser.denoising(q_i_dn.unsqueeze(0),self.denoise_timestep)
                q_i_dn=q_i_dn.squeeze()
                
                ## Calcluate MSEs
                mses_dn[cc]=(criterion(q_i_true,q_i_dn))
                mses[cc]=(criterion(q_i_true,q_i))
                mses_0[cc]=(criterion(q_i_true,init))
                cc+=1
                
            plt.title("MSE(truth,emulator) in blue, MSE(truth at t=0, truth at t=i) in red")
            plt.plot(times,mses,color="blue",label="Emulator MSE wrt truth",alpha=0.2)
            plt.plot(times,mses_0,color="red",label="True MSE wrt t=0",alpha=0.2)
            plt.plot(times,mses_dn,color="orange",label="True MSE wrt t=0",alpha=0.4)
            plt.yscale("log")
            plt.ylim(1e-3,5e0)
            plt.xlabel("timestep")
            plt.ylabel("MSE")

        return fig

    def normalise(self,q):
        ## Map from physical to normalised space using the factors used to train the network
        ## Normalise each field individually, then cat arrays back to shape appropriate for a torch model
        x_upper = transforms.normalise_field(q[0],self.network.config["q_mean_upper"],self.network.config["q_std_upper"])
        x_lower = transforms.normalise_field(q[1],self.network.config["q_mean_lower"],self.network.config["q_std_lower"])
        x = torch.stack((x_upper,x_lower),dim=0)
        return x
        