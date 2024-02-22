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

        #x=torch.tensor(q_i,device=self.device)
        ## Map from physical to normalised space using the factors used to train the network
        ## Normalise each field individually, then cat arrays back to shape appropriate for a torch model
        x = self.denorm(q_i)
        x = x.unsqueeze(0)

        x.to
        x=self.network(x)

        ## Map back from normalised space to physical units
        q_upper=transforms.denormalise_field(x[:,0,:,:],self.network.config["q_mean_upper"],self.network.config["q_std_upper"])
        q_lower=transforms.denormalise_field(x[:,1,:,:],self.network.config["q_mean_lower"],self.network.config["q_std_lower"])

        ## Set zero mean
        q_upper=q_upper-torch.mean(q_upper)
        q_lower=q_lower-torch.mean(q_lower)

        ## Reshape to match pyqg dimensions, and cast to numpy array
        q_i_dt=torch.cat((q_upper,q_lower))
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
            q_i=torch.tensor(ds.q[aa,0].to_numpy(),device=self.device,dtype=torch.float32)
            ## Index counter for loss arrays
            cc=0
            init=self.denorm(torch.tensor(ds.q[aa,0].values))
            for bb in range(self.network.config["increment"],ds.q.shape[1],self.network.config["increment"]):
                q_i_dt=self._get_next_step(q_i)
                q_i_pred=self.denorm(q_i_dt)
                q_i_true=self.denorm(torch.tensor(ds.q[aa,bb].values,device=self.device))
                mses[cc]=(criterion(q_i_true,q_i_pred))
                mses_0[cc]=(criterion(q_i_true,init))
                q_i=q_i_dt
                cc+=1
                
            plt.title("MSE(truth,emulator), for 20 trajectories")
            plt.plot(times,mses,color="blue",label="Emulator MSE wrt truth",alpha=0.2)

            plt.plot(times,mses_0,color="red",label="True MSE wrt t=0",alpha=0.2)
            plt.yscale("log")
            plt.ylim(1e-3,5e0)
            plt.xlabel("timestep")
            plt.ylabel("MSE")

        return fig

    def denorm(self,q):
        ## Map from physical to normalised space using the factors used to train the network
        ## Normalise each field individually, then cat arrays back to shape appropriate for a torch model
        x_upper = transforms.normalise_field(q[0],self.network.config["q_mean_upper"],self.network.config["q_std_upper"])
        x_lower = transforms.normalise_field(q[1],self.network.config["q_mean_lower"],self.network.config["q_std_lower"])
        x = torch.stack((x_upper,x_lower),dim=0)
        return x
