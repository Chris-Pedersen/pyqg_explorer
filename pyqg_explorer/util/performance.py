import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import xarray as xr
import pickle
import pyqg

import matplotlib.animation as animation
from IPython.display import HTML

import pyqg_explorer.util.powerspec as powerspec
import pyqg_explorer.parameterizations.parameterizations as parameterizations
import pyqg_explorer.generate_datasets as generate_datasets
import pyqg_explorer.util.transforms as transforms
import cmocean


class EmulatorPerformance():
    """ Object to store performance tests relevant to neural emulators """
    def __init__(self,network,valid_loader,threshold,rollout=False):
        """ network:  Torch model we want to test. Assuming this is a model for the subgrid forcing
            valid_loader: torch dataloader with the validation set
                          NB we are assuming this is an EmulatorForcingDataset, where the
                          true subgrid forcing is in the x[:,2:4,:,:] indices
            threshold: The number of data samples at which to stop iterating dataloader. We take a subsample
                       in order to avoid running out of memory, and have demonstrated that a few thousand samples
                       is enough to estimate offline metrics
                       
                       
            TO ADD: an assertion clause to make sure the random seed in the model and dataset are the same, to ensure we are
                    predicting on unseen data """
        
        self.network=network
        self.x_np=[]
        self.y_true=[] ## Let's store the i+1 field
        self.y_pred=[] ## for both true and predicted
        self.rollout=rollout
        
        if valid_loader is not None:
            self._populate_fields(valid_loader,threshold)

    def _populate_fields(self,valid_loader,threshold):
        count=0
        if self.rollout==False:
            ## Cache x, true y and predicted y values that we will use to guage offline performance
            for data in valid_loader:
                x=data[0]
                y=data[1]
                count+=x.shape[0]
                self.x_np.append(x)
                self.y_true.append(y)
                self.y_pred.append(self.network(x)+x[:,0:2,:,:]) ## y pred should now be the same quantity, q_i+dt
                if count>threshold:
                    break
        else:
            ## Cache x, true y and predicted y values that we will use to guage offline performance
            for data in valid_loader:
                x=data
                count+=x.shape[0]
                self.x_np.append(x[:,:,0,:,:])
                self.y_true.append(x[:,:,1,:,:])
                self.y_pred.append(self.network(x[:,:,0,:,:])+x[:,:,0,:,:]) ## y pred should now be the same quantity, q_i+dt
                if count>threshold:
                    break
            
        self.x_np=torch.vstack(self.x_np).detach().numpy()
        self.y_true=torch.vstack(self.y_true).detach().numpy()
        self.y_pred=torch.vstack(self.y_pred).detach().numpy()

        ## Save R2, since we basically get it for free at this point
        self.r2_upper=r2_score(self.y_true[:,0,:,:].flatten(),self.y_pred[:,0,:,:].flatten())
        self.r2_lower=r2_score(self.y_true[:,1,:,:].flatten(),self.y_pred[:,1,:,:].flatten())
        
    def get_distribution_2d(self):
        """ Plot histograms of the true and predicted subgrid forcing """
        fig, axs = plt.subplots(1, 2,figsize=(11,4))
        axs[0].set_title(r"Upper layer: $R^2$=%.5f" % self.r2_upper)
        axs[1].set_title(r"Lower layer: $R^2$=%.5f" % self.r2_lower)
        line=np.linspace(-4,4,100)
        axs[0].plot(line,line,linestyle="dashed",color="gray",alpha=0.5)
        ax=axs[0].hist2d(self.y_true[:,0,:,:].flatten()-self.x_np[:,0,:,:].flatten(),self.y_pred[:,0,:,:].flatten()-self.x_np[:,0,:,:].flatten(),bins=100,range=[[-0.1,0.1],[-0.1,0.1]],cmap='RdPu');
        fig.colorbar(ax[3], ax=axs[0])
        axs[1].plot(line,line,linestyle="dashed",color="gray",alpha=0.5)
        ax=axs[1].hist2d(self.y_true[:,1,:,:].flatten()-self.x_np[:,1,:,:].flatten(),self.y_pred[:,1,:,:].flatten()-self.x_np[:,1,:,:].flatten(),bins=100,range=[[-0.1,0.1],[-0.1,0.1]],cmap='RdPu');
        fig.colorbar(ax[3], ax=axs[1])
        return fig
    
    def get_correlation(self,time_rollout=1000):
        """ Plot correlation over time with respect to a standard test simulation """
        
        timesteps=np.arange(0,time_rollout+0.01,self.network.config["time_horizon"],dtype=int)

        ds=xr.load_dataset("/scratch/cp3759/pyqg_data/sims/animation_sims/lowres_3k.nc")
        ds=ds.q

        q_i_pred=ds[0].to_numpy()

        correlation_upper=[]
        correlation_lower=[]

        autocorrelation_upper=[]
        autocorrelation_lower=[]

        for aa in timesteps:
            ## Get correlation between truth and emulator
            correlation_upper.append(pearsonr(q_i_pred[0].flatten(),ds[aa,0].to_numpy().flatten())[0])
            correlation_lower.append(pearsonr(q_i_pred[1].flatten(),ds[aa,1].to_numpy().flatten())[0])

            autocorrelation_upper.append(pearsonr(ds[aa,0].to_numpy().flatten(),ds[0,0].to_numpy().flatten())[0])
            autocorrelation_lower.append(pearsonr(ds[aa,1].to_numpy().flatten(),ds[0,1].to_numpy().flatten())[0])

            x=torch.tensor(q_i_pred).float()

            x_upper = transforms.normalise_field(x[0],self.network.config["q_mean_upper"],self.network.config["q_std_upper"])
            x_lower = transforms.normalise_field(x[1],self.network.config["q_mean_lower"],self.network.config["q_std_lower"])
            x_denorm = torch.stack((x_upper,x_lower),dim=0).unsqueeze(0)

            x_dt=self.network(x_denorm)

            ## Map back from normalised space to physical units
            q_upper=transforms.denormalise_field(x_dt[:,0,:,:],self.network.config["q_mean_upper"],self.network.config["q_std_upper"])
            q_lower=transforms.denormalise_field(x_dt[:,1,:,:],self.network.config["q_mean_lower"],self.network.config["q_std_lower"])

            normalise=False
            if normalise==True:
                q_upper=q_upper-torch.mean(q_upper)
                q_lower=q_lower-torch.mean(q_lower)

            q_i_pred=(torch.cat((q_upper,q_lower))+x.squeeze()).detach().numpy().astype(np.double)

        fig=plt.figure()
        plt.title("Correlation wrt test simulation")
        plt.plot(timesteps,correlation_upper,label="upper",color="blue",linestyle="solid")
        plt.plot(timesteps,correlation_lower,label="lower",color="blue",linestyle="dashed")
        plt.plot(timesteps,autocorrelation_upper,label="upper auto",color="black",linestyle="solid")
        plt.plot(timesteps,autocorrelation_lower,label="lower auto",color="black",linestyle="dashed")
        plt.xlabel("Time (hours)")
        plt.legend()
        return fig
        
    def get_fields(self,map_index=None):
        """ For a single data sample, plot the input field, target output, predicted output, and diff between the two """
        
        ## Chose random index unless one is provided
        if map_index is None:
            map_index=np.random.randint(len(self.x_np)-1)
    
        fig, axs = plt.subplots(2, 4,figsize=(15,6))
        image=self.x_np[map_index][0]
        limit=np.max(np.abs(image))
        ax=axs[0][0].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit,interpolation='none')
        fig.colorbar(ax, ax=axs[0][0])
        axs[0][0].set_xticks([]); axs[0][0].set_yticks([])
        axs[0][0].set_title("q_i")

        image=self.y_true[map_index][0]-self.x_np[map_index][0]
        limit=np.max(np.abs(image))
        ax=axs[0][1].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[0][1])
        axs[0][1].set_xticks([]); axs[0][1].set_yticks([])
        axs[0][1].set_title("true res")

        image=self.y_pred[map_index][0]-self.x_np[map_index][0]
        limit=np.max(np.abs(image))
        ax=axs[0][2].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[0][2])
        axs[0][2].set_xticks([]); axs[0][2].set_yticks([])
        axs[0][2].set_title("pred res")

        image=self.y_true[map_index][0]-self.y_pred[map_index][0]
        limit=np.max(np.abs(image))
        ax=axs[0][3].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[0][3])
        axs[0][3].set_xticks([]); axs[0][3].set_yticks([])
        axs[0][3].set_title("diff")
        fig.tight_layout()

        image=self.x_np[map_index][1]
        limit=np.max(np.abs(image))
        ax=axs[1][0].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[1][0])
        axs[1][0].set_xticks([]); axs[1][0].set_yticks([])

        image=self.y_true[map_index][1]-self.x_np[map_index][1]
        limit=np.max(np.abs(image))
        ax=axs[1][1].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[1][1])
        axs[1][1].set_xticks([]); axs[1][1].set_yticks([])

        image=self.y_pred[map_index][1]-self.x_np[map_index][1]
        limit=np.max(np.abs(image))
        ax=axs[1][2].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[1][2])
        axs[1][2].set_xticks([]); axs[1][2].set_yticks([])

        image=self.y_true[map_index][1]-self.y_pred[map_index][1]
        limit=np.max(np.abs(image))
        ax=axs[1][3].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[1][3])
        axs[1][3].set_xticks([]); axs[1][3].set_yticks([])
        fig.tight_layout()
        
        return fig


class ParameterizationPerformance():
    """ Object to store performance tests relevant to neural parameterizations """
    def __init__(self,network,valid_loader,threshold):
        """ network:  Torch model we want to test. Assuming this is a model for the subgrid forcing
            valid_loader: torch dataloader with the validation set
                          NB we are assuming this is an EmulatorForcingDataset, where the
                          true subgrid forcing is in the x[:,2:4,:,:] indices
            threshold: The number of data samples at which to stop iterating dataloader. We take a subsample
                       in order to avoid running out of memory, and have demonstrated that a few thousand samples
                       is enough to estimate offline metrics
                       
                       
            TO ADD: an assertion clause to make sure the random seed in the model and dataset are the same, to ensure we are
                    predicting on unseen data """
        
        self.network=network
        self.x_np=[]
        self.y_true=[] ## True subgrid forcing
        self.y_pred=[] ## Predicted subgrid forcing
        
        count=0
        ## Cache x, true y and predicted y values that we will use to guage offline performance
        for data in valid_loader:
            x=data[0]
            y=data[1]
            count+=x.shape[0]
            if x.shape[1]>2:
                self.x_np.append(x[:,0:2,:])   
                self.y_true.append(x[:,2:4,:]) 
                self.y_pred.append(self.network(x[:,0:2,:]))
            else:
                self.x_np.append(x)
                self.y_true.append(y)
                self.y_pred.append(self.network(x))
            if count>threshold:
                break
        self.x_np=torch.vstack(self.x_np).detach().numpy()
        self.y_true=torch.vstack(self.y_true).detach().numpy()
        self.y_pred=torch.vstack(self.y_pred).detach().numpy() ## Do we really need to save this?
        
        ## Save R2, since we basically get it for free at this point
        self.r2_upper=r2_score(self.y_true[:,0,:,:].flatten(),self.y_pred[:,0,:,:].flatten())
        self.r2_lower=r2_score(self.y_true[:,1,:,:].flatten(),self.y_pred[:,1,:,:].flatten())
    
    ## Offline tests
    def get_power_spectrum(self):
        """ Plot power spectra of true and predicted subgrid forcing """
        power_upper_true=[]
        power_lower_true=[]
        power_upper_pred=[]
        power_lower_pred=[]

        for aa in range(len(self.x_np)):
            power_upper_true.append(powerspec.get_power_spectrum(self.y_true[aa][0]))
            power_lower_true.append(powerspec.get_power_spectrum(self.y_true[aa][1]))
            power_upper_pred.append(powerspec.get_power_spectrum(self.y_pred[aa][0]))
            power_lower_pred.append(powerspec.get_power_spectrum(self.y_pred[aa][1]))

        power_upper_true=np.mean(np.stack(power_upper_true,axis=1),axis=1)
        power_lower_true=np.mean(np.stack(power_lower_true,axis=1),axis=1)
        power_upper_pred=np.mean(np.stack(power_upper_pred,axis=1),axis=1)
        power_lower_pred=np.mean(np.stack(power_lower_pred,axis=1),axis=1)
        
        fig, axs = plt.subplots(1, 2,figsize=(11,4))
        axs[0].set_title(r"Upper layer: $R^2$=%.2f" % self.r2_upper)
        axs[1].set_title(r"Lower layer: $R^2$=%.2f" % self.r2_lower)
        axs[0].loglog(power_upper_true,label="True")
        axs[0].loglog(power_upper_pred,label="Predicted")
        axs[1].loglog(power_lower_true)
        axs[1].loglog(power_lower_pred)
        axs[0].legend()
        return fig
    
    def get_R2(self):
        """ Return estimated values for R2 in upper and lower layers """
        return self.r2_upper,self.r2_lower
        
    def get_distribution(self):
        """ Plot histograms of the true and predicted subgrid forcing """
        fig, axs = plt.subplots(1, 2,figsize=(11,4))
        axs[0].set_title(r"Upper layer: $R^2$=%.2f" % self.r2_upper)
        axs[1].set_title(r"Lower layer: $R^2$=%.2f" % self.r2_lower)
        axs[0].hist(self.x_np[:,2,:,:].flatten(),bins=200,density=True,alpha=0.5,label="True");
        axs[0].hist(self.y_pred[:,0,:,:].flatten(),bins=200,density=True,alpha=0.5,label="Predicted");
        axs[1].hist(self.x_np[:,3,:,:].flatten(),bins=200,density=True,alpha=0.5);
        axs[1].hist(self.y_pred[:,1,:,:].flatten(),bins=200,density=True,alpha=0.5);
        axs[0].legend()
        return fig

    def get_distribution_2d(self):
        """ Plot histograms of the true and predicted subgrid forcing """
        fig, axs = plt.subplots(1, 2,figsize=(11,4))
        axs[0].set_title(r"Upper layer: $R^2$=%.2f" % self.r2_upper)
        axs[1].set_title(r"Lower layer: $R^2$=%.2f" % self.r2_lower)
        line=np.linspace(-4,4,100)
        axs[0].plot(line,line,linestyle="dashed",color="gray",alpha=0.5)
        ax=axs[0].hist2d(self.y_true[:,0,:,:].flatten(),self.y_pred[:,0,:,:].flatten(),bins=100,range=[[-4,4],[-4,4]],cmap='RdPu');
        fig.colorbar(ax[3], ax=axs[0])
        axs[1].plot(line,line,linestyle="dashed",color="gray",alpha=0.5)
        ax=axs[1].hist2d(self.y_true[:,1,:,:].flatten(),self.y_pred[:,1,:,:].flatten(),bins=100,range=[[-4,4],[-4,4]],cmap='RdPu');
        fig.colorbar(ax[3], ax=axs[1])
        return fig

    def subgrid_energy(self):
        """ Plot contribution of energy flux from subgrid forcing model. Can be calculated offline. We use eqs
            E2-E4 of arxiv.org/abs/2302.07984 to estimate the energy contribution from the subgrid model, which
            comes from Re(conj(fft(psi)))*fft(S)

            We use the stored normalisation factors to convert the q and s fields back into physical units.
            Then reconstruct streamfunction, psi, using pyqg inversion method. Then combine this with the subgrid
            forcing field to determine the energy contribution from the subgrid model.

            Currently this is hardcoded to work with a 64x64 coarse resolution system, as this is all I am working with
            for now. Have added an assertion to this effect - if we work with other resolutions will have to extend this
            to work dynamically with different resolutions.

            """
        assert self.x_np.shape[-1]==64, "Only works with 64x64 coarse system so far"
        
        ## Denorm to physical units using normalisation factors that the model was trained using
        q_upper=(self.x_np[:,0]*self.network.config["q_std_upper"].numpy())+self.network.config["q_mean_upper"].numpy()
        q_lower=(self.x_np[:,1]*self.network.config["q_std_lower"].numpy())+self.network.config["q_mean_lower"].numpy()

        s_true_upper=(self.y_true[:,0]*self.network.config["s_std_upper"].numpy())+self.network.config["s_mean_upper"].numpy()
        s_true_lower=(self.y_true[:,1]*self.network.config["s_std_lower"].numpy())+self.network.config["s_mean_lower"].numpy()
        
        s_pred_upper=(self.y_pred[:,0]*self.network.config["s_std_upper"].numpy())+self.network.config["s_mean_upper"].numpy()
        s_pred_lower=(self.y_pred[:,1]*self.network.config["s_std_lower"].numpy())+self.network.config["s_mean_lower"].numpy()
        
        self.denormed_q=np.stack((q_upper,q_lower),axis=1).astype(np.float64)
        self.denormed_s_true=np.stack((s_true_upper,s_true_lower),axis=1).astype(np.float64)
        self.denormed_s_pred=np.stack((s_pred_upper,s_pred_lower),axis=1).astype(np.float64)

        ## Use pyqg inversion function to get streamfunction, psi
        ## Hardcode to 64x64 images for now
        self.psi=np.empty((len(self.denormed_q),2,64,64))
        self.de_dt_true=np.empty((len(self.denormed_q),2,32))
        self.de_dt_pred=np.empty((len(self.denormed_q),2,32))
        
        m = pyqg.QGModel(log_level = 0)
        for aa in range(len(self.denormed_q)):
            m.q = self.denormed_q[aa]
            m._invert()
            self.psi[aa]=(m.to_dataset().p).to_numpy()
            
            ## Take FFT of stream function and subgrid forcing
            fftpsi=np.fft.rfftn(self.psi[aa], axes=(-2,-1))/(64**2)
            fftsubgrid_true=np.fft.rfftn(self.denormed_s_true[aa], axes=(-2,-1))/(64**2)
            fftsubgrid_pred=np.fft.rfftn(self.denormed_s_pred[aa], axes=(-2,-1))/(64**2)

            ## Combine to form dE/dt
            dt_true=np.real(np.conj(fftpsi)) * fftsubgrid_true
            dt_pred=np.real(np.conj(fftpsi)) * fftsubgrid_pred
            
            ## Get isotropic spectra for upper and lower layers
            spectra_true=[]
            spectra_pred=[]
            
            ## Do upper and lower layers
            for z in [0,1]:
                k, sp = pyqg.diagnostic_tools.calc_ispec(m, dt_true[z,:,:], averaging=False, truncate=False)
                spectra_true.append(-sp)
                k, sp = pyqg.diagnostic_tools.calc_ispec(m, dt_pred[z,:,:], averaging=False, truncate=False)
                spectra_pred.append(-sp)
            
            ## Convert to array
            self.k=k
            self.de_dt_true[aa]=np.array(spectra_true)
            self.de_dt_pred[aa]=np.array(spectra_pred)

        
        
        fig, axs = plt.subplots(1, 2,figsize=(13,5))

        axs[0].set_title("Energy contribution from subgrid forcing, upper layer")
        axs[0].plot(self.k,np.mean(self.de_dt_pred,axis=0)[0],label="ML model")
        axs[0].plot(self.k,np.mean(self.de_dt_true,axis=0)[0],label="true")
        axs[0].legend()
        axs[0].set_ylabel(r"$\frac{dE}{dt}$ by subgrid forcing")
        axs[0].set_xlabel("wavenumber (1/m)")
        axs[1].set_title("Lower layer")
        axs[1].plot(self.k,np.mean(self.de_dt_pred,axis=0)[1],label="ML model")
        axs[1].plot(self.k,np.mean(self.de_dt_true,axis=0)[1],label="true")
        axs[1].set_xlabel("wavenumber (1/m)")
        plt.tight_layout()
        
        return fig
        
    def get_fields(self,map_index=None):
        """ For a single data sample, plot the input field, target output, predicted output, and diff between the two """
        
        ## Chose random index unless one is provided
        if map_index is None:
            map_index=np.random.randint(len(self.x_np)-1)
    
        fig, axs = plt.subplots(2, 4,figsize=(15,6))
        image=self.x_np[map_index][0]
        limit=np.max(np.abs(image))
        ax=axs[0][0].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit,interpolation='none')
        fig.colorbar(ax, ax=axs[0][0])
        axs[0][0].set_xticks([]); axs[0][0].set_yticks([])
        axs[0][0].set_title("PV field")

        image=self.y_true[map_index][0]
        limit=np.max(np.abs(image))
        ax=axs[0][1].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[0][1])
        axs[0][1].set_xticks([]); axs[0][1].set_yticks([])
        axs[0][1].set_title("True forcing")

        image=self.y_pred[map_index][0]
        limit=np.max(np.abs(image))
        ax=axs[0][2].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[0][2])
        axs[0][2].set_xticks([]); axs[0][2].set_yticks([])
        axs[0][2].set_title("Forcing from CNN")

        image=self.y_true[map_index][0]-self.y_pred[map_index][0]
        limit=np.max(np.abs(image))
        ax=axs[0][3].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[0][3])
        axs[0][3].set_xticks([]); axs[0][3].set_yticks([])
        axs[0][3].set_title("True forcing-CNN forcing")
        fig.tight_layout()

        image=self.x_np[map_index][1]
        limit=np.max(np.abs(image))
        ax=axs[1][0].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[1][0])
        axs[1][0].set_xticks([]); axs[1][0].set_yticks([])

        image=self.y_true[map_index][1]
        limit=np.max(np.abs(image))
        ax=axs[1][1].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[1][1])
        axs[1][1].set_xticks([]); axs[1][1].set_yticks([])

        image=self.y_pred[map_index][1]
        limit=np.max(np.abs(image))
        ax=axs[1][2].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[1][2])
        axs[1][2].set_xticks([]); axs[1][2].set_yticks([])

        image=self.y_true[map_index][1]-self.y_pred[map_index][1]
        limit=np.max(np.abs(image))
        ax=axs[1][3].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[1][3])
        axs[1][3].set_xticks([]); axs[1][3].set_yticks([])
        fig.tight_layout()
        
        return fig

    ## Online tests
    def online_comparison(self):
        def KE(ds_test):
            return (ds_test.u**2 + ds_test.v**2) * 0.5

        def get_ke_time(ds_test):
            ke=KE(ds_test)
            ke_array=[]
            for snaps in ke:
                ke_array.append(ds_test.attrs['pyqg:L']*np.sum(snaps.data)/(ds_test.attrs['pyqg:nx'])**2)
            return ke_array
        parameterization=parameterizations.Parameterization(self.network)
        ds = generate_datasets.generate_dataset(parameterization=parameterization)
        low_res=xr.open_dataset('/scratch/cp3759/pyqg_data/sims/online_test_reference_sims/test_sim_641.nc')
        high_res=xr.open_dataset('/scratch/cp3759/pyqg_data/sims/online_test_reference_sims/test_sim_2561.nc')
        theta_only=xr.open_dataset('/scratch/cp3759/pyqg_data/sims/online_test_reference_sims/test_sim_64_thetaonly1.nc')

        fig, axs = plt.subplots(1, 3,figsize=(17,5))
        axs[0].set_title("KE spectrum")
        axs[0].loglog(high_res.ispec_k.data,high_res.ispec_KEspec_avegd.data,label="high res (256)",color="black",lw=3)
        axs[0].loglog(low_res.ispec_k.data,low_res.ispec_KEspec_avegd.data,label="no param (64)",linestyle="dashed")
        axs[0].loglog(theta_only.ispec_k.data,theta_only.ispec_KEspec_avegd.data,label="theta only CNN (64)",linestyle="dashed")
        axs[0].loglog(ds.ispec_k.data,ds.ispec_KEspec_avegd.data,label="This tested model!",linestyle="-.",lw=2)
        axs[0].set_ylim(1e-2,1e2)
        axs[0].set_xlim(5e-6,2e-4)
        axs[0].legend()

        axs[1].set_title("Energy transfer")
        axs[1].plot(high_res.ispec_k.data,high_res.ispec_energy_transfer.data,label="high res (256)",lw=3,color="black")
        axs[1].plot(low_res.ispec_k.data,low_res.ispec_energy_transfer.data,label="no param (64)",linestyle="dashed")
        axs[1].plot(theta_only.ispec_k.data,theta_only.ispec_energy_transfer.data,label="theta only CNN (64)",linestyle="dashed")
        axs[1].plot(ds.ispec_k.data,ds.ispec_energy_transfer.data,label="This tested model!",linestyle="-.",lw=2)
        axs[1].set_xscale("log")
        axs[1].set_xlim(4e-6,2e-4)

        ## Hardcoded for now - eventually want to have this read from the simulation attrs
        x_years=np.linspace(0,10,87)

        axs[2].set_title("KE over time (years)")
        axs[2].plot(x_years,get_ke_time(high_res),label="high res (256)",lw=3,color="black")
        axs[2].plot(x_years,get_ke_time(low_res),label="no param (64)",linestyle="dashed")
        axs[2].plot(x_years,get_ke_time(theta_only),label="theta only CNN (64)",linestyle="dashed")
        axs[2].plot(x_years,get_ke_time(ds),label="This tested model!",linestyle="-.",lw=2)
        return fig


class EmulatorAnimation():
    def __init__(self,q_ds,model,fps=10,nSteps=1000,normalise=True):
        self.q_ds=q_ds
        self.model=model
        self.fps = fps
        self.nSteps = nSteps
        self.nFrames = int(self.nSteps/self.model.config["time_horizon"])
        self.q_i_pred=q_ds[0].data
        self.normalise=normalise
        self.mse=[]
        self.correlation_upper=[]
        self.correlation_lower=[]
        self.autocorrelation_upper=[]
        self.autocorrelation_lower=[]
        self.criterion=nn.MSELoss()
        self.times=np.arange(0,self.nFrames*self.model.config["time_horizon"]+0.01,self.model.config["time_horizon"])
        
    def _push_forward(self):
        """ Update predicted q by one emulator pass """
        
        ## Convert q to standardised q
        x=torch.tensor(self.q_i_pred).float()
        ## Map from physical to normalised space using the factors used to train the network
        ## Normalise each field individually, then cat arrays back to shape appropriate for a torch model
        x_upper = transforms.normalise_field(x[0],self.model.config["q_mean_upper"],self.model.config["q_std_upper"])
        x_lower = transforms.normalise_field(x[1],self.model.config["q_mean_lower"],self.model.config["q_std_lower"])
        x = torch.stack((x_upper,x_lower),dim=0).unsqueeze(0)

        x=self.model(x)

        ## Map back from normalised space to physical units
        q_upper=transforms.denormalise_field(x[:,0,:,:],self.model.config["q_mean_upper"],self.model.config["q_std_upper"])
        q_lower=transforms.denormalise_field(x[:,1,:,:],self.model.config["q_mean_lower"],self.model.config["q_std_lower"])
        
        if self.normalise==True:
            q_upper=q_upper-torch.mean(q_upper)
            q_lower=q_lower-torch.mean(q_lower)

        ## Reshape to match pyqg dimensions, and cast to numpy array
        q_i_dt=torch.cat((q_upper,q_lower)).detach().numpy().astype(np.double)
                    
        self.q_i_pred=self.q_i_pred+q_i_dt
        
        self.correlation_upper.append(pearsonr(self.q_i_pred[0].flatten(),self.q_ds[self.ds_i,0].to_numpy().flatten())[0])
        self.correlation_lower.append(pearsonr(self.q_i_pred[1].flatten(),self.q_ds[self.ds_i,1].to_numpy().flatten())[0])
        self.autocorrelation_upper.append(pearsonr(self.q_ds[0,0].to_numpy().flatten(),self.q_ds[self.ds_i,0].to_numpy().flatten())[0])
        self.autocorrelation_lower.append(pearsonr(self.q_ds[0,1].to_numpy().flatten(),self.q_ds[self.ds_i,1].to_numpy().flatten())[0])
        self.mse.append(self.criterion(torch.tensor(self.q_i_pred),torch.tensor(self.q_ds[self.ds_i].to_numpy())))
        
        return
    
    def animate(self):
        fig, axs = plt.subplots(2, 4,figsize=(14,6))
        self.ax1=axs[0][0].imshow(self.q_ds[0].data[0], cmap=cmocean.cm.balance)
        fig.colorbar(self.ax1, ax=axs[0][0])
        axs[0][0].set_xticks([]); axs[0][0].set_yticks([])
        axs[0][0].set_title("PyQG")

        self.ax2=axs[0][1].imshow(self.q_ds[0].data[0], cmap=cmocean.cm.balance, interpolation='none')
        fig.colorbar(self.ax2, ax=axs[0][1])
        axs[0][1].set_xticks([]); axs[0][1].set_yticks([])
        axs[0][1].set_title("Emulator")

        self.ax3=axs[0][2].imshow(self.q_ds[0].data[0], cmap=cmocean.cm.balance, interpolation='none')
        fig.colorbar(self.ax3, ax=axs[0][2])
        axs[0][2].set_xticks([]); axs[0][2].set_yticks([])
        axs[0][2].set_title("Residuals")

        fig.tight_layout()

        self.ax4=axs[1][0].imshow(self.q_ds[0].data[1], cmap=cmocean.cm.balance)
        fig.colorbar(self.ax4, ax=axs[1][0])
        axs[1][0].set_xticks([]); axs[1][0].set_yticks([])

        self.ax5=axs[1][1].imshow(self.q_ds[0].data[1], cmap=cmocean.cm.balance, interpolation='none')
        fig.colorbar(self.ax5, ax=axs[1][1])
        axs[1][1].set_xticks([]); axs[1][1].set_yticks([])

        self.ax6=axs[1][2].imshow(self.q_ds[0].data[1], cmap=cmocean.cm.balance, interpolation='none')
        cb6=fig.colorbar(self.ax6, ax=axs[1][2])
        axs[1][2].set_xticks([]); axs[1][2].set_yticks([])
        
        ## Time evol metrics
        axs[0][3].set_title("Correlation")
        self.ax7=[axs[0][3].plot(-1),axs[0][3].plot(-1)]
        axs[0][3].set_ylim(0,1)
        axs[0][3].set_xlim(0,self.times[-1])
        
        self.ax8=[axs[1][3].plot(-1),axs[1][3].plot(-1)]
        axs[1][3].set_ylim(0,1)
        axs[1][3].set_xlim(0,self.times[-1])
        
        self.time_text=axs[0][2].text(-20,-20,"")
        
        fig.tight_layout()
        
        anim = animation.FuncAnimation(
                                       fig, 
                                       self.animate_func, 
                                       frames = self.nFrames,
                                       interval = 1000 / self.fps, # in ms
                                       )
        plt.close()
        
        return HTML(anim.to_html5_video())
        
    def animate_func(self,i):
        if i % self.fps == 0:
            print( '.', end ='' )
            
        self.i=i
        self.ds_i=i*self.model.config["time_horizon"]
        self.time_text.set_text("%d timesteps" % (i*self.model.config["time_horizon"]))
    
        ## Set image and colorbar for each panel
        image=self.q_ds[self.ds_i].data[0]
        self.ax1.set_array(image)
        self.ax1.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.q_i_pred[0]
        self.ax2.set_array(image)
        self.ax2.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.q_i_pred[0]-self.q_ds[self.ds_i].data[0]
        self.ax3.set_array(image)
        self.ax3.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.q_ds[i].data[1]
        self.ax4.set_array(image)
        self.ax4.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.q_i_pred[1]
        self.ax5.set_array(image)
        self.ax5.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.q_i_pred[1]-self.q_ds[self.ds_i].data[1]
        self.ax6.set_array(image)
        self.ax6.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
 
        self.ax7[0][0].set_xdata(np.array(self.times[0:len(self.correlation_upper)]))
        self.ax7[0][0].set_ydata(np.array(self.correlation_upper))
        
        self.ax7[1][0].set_xdata(np.array(self.times[0:len(self.autocorrelation_upper)]))
        self.ax7[1][0].set_ydata(np.array(self.autocorrelation_upper))
        
        self.ax8[0][0].set_xdata(np.array(self.times[0:len(self.correlation_lower)]))
        self.ax8[0][0].set_ydata(np.array(self.correlation_lower))
                  
        self.ax8[1][0].set_xdata(np.array(self.times[0:len(self.autocorrelation_lower)]))
        self.ax8[1][0].set_ydata(np.array(self.autocorrelation_lower))
        
        self._push_forward()
        
        return 
        
#############################################################################################
###### Functions related to stability tests (i.e. plotting kinetic energy accumulation ######
###### over time for various different sims)                                           ######
#############################################################################################

def KE_func(ds_test):
    """ Get KE from the velocities stored in a dataset """
    
    return (ds_test.u**2 + ds_test.v**2) * 0.5

def get_ke_time(ds_test):
    """ Get an array of kinetic energy over time from a dataset"""
    
    ke=KE_func(ds_test)
    ke_array=[]
    for snaps in ke:
        ke_array.append(ds_test.attrs['pyqg:L']*np.sum(snaps.data)/(ds_test.attrs['pyqg:nx'])**2)
        
    return ke_array

def load_ensembles(ensemble_path):
    """ Loop over ensemble members and alphas to build a grid of kinetic energy over time. Assumes we have
    50 simulations in each ensemble, and alphas range from [1,5,10,15] """
    
    grid_ke=np.zeros((4,50,173))
    spectral_energy=np.zeros((4,50,23))
    for aa,alpha in enumerate([1,5,10,15]):
        for bb in range(50):
            data=xr.open_dataset(ensemble_path+"/alpha_"+str(alpha)+"_run_"+str(bb+1)+".nc")
            grid_ke[aa][bb]=get_ke_time(data)
            spectral_energy[aa][bb]=data.ispec_energy_transfer.data
            
    return grid_ke,spectral_energy

def plot_KE_alphas(sim_path,label=r"$\mathcal{L}_\theta+\mathcal{L}_\beta$"):
    """ Function to read a directory of sims with varying alphas, and plot both the spectral energy transfer
        and the KE accumulation as a function of time for the simulations in question, alongside some reference
        baselines. We build the data grid using misc.performance.load_ensembles, which assumes we have 50 simulations
        for each alpha, and the alphas range [1,5,10,15].
        
        sim_path: path to the ensemble of sims
        label: custom label for the KE(time) plot for these sims
        """
    
    ## Load xarray data for each snapshot, and extract both the spectra energy and KE(time) for all different alphas
    alpha_data,spectral_data=load_ensembles(sim_path)
    
    ## Assuming our test sims are 64^2, use a standardised wavenumber array
    spectral_k=np.array([4.44288294e-06, 1.33286488e-05, 2.22144147e-05, 3.11001806e-05,
           3.99859464e-05, 4.88717123e-05, 5.77574782e-05, 6.66432441e-05,
           7.55290099e-05, 8.44147758e-05, 9.33005417e-05, 1.02186308e-04,
           1.11072073e-04, 1.19957839e-04, 1.28843605e-04, 1.37729371e-04,
           1.46615137e-04, 1.55500903e-04, 1.64386669e-04, 1.73272435e-04,
           1.82158200e-04, 1.91043966e-04, 1.99929732e-04])
    model_time=np.linspace(0,20,87*2-1)
    
    ## Load reference comparison (hardcoded to a pure-offline parameterised sim for now)
    with open("/scratch/cp3759/pyqg_data/sims/KE_accumulation/offline_only_test/cached_data.p", "rb") as input_file:
        cached_data = pickle.load(input_file)
    reference_alpha=cached_data[0]
    reference_spectral=cached_data[1]
    
    ## Load low res and high res for spectra
    low_res=xr.open_dataset('/scratch/cp3759/pyqg_data/sims/online_test_reference_sims/test_sim_641.nc')
    high_res=xr.open_dataset('/scratch/cp3759/pyqg_data/sims/online_test_reference_sims/test_sim_2561.nc')
    
    ## Extract KE(time) data for high res, alpha=1
    hires=np.zeros((50,173))
    for aa in range(50):
        entry_string="/scratch/cp3759/pyqg_data/sims/KE_accumulation/hires/highre%ds_1" % (aa+1)
        data=xr.open_dataset(entry_string)
        hires[aa]=get_ke_time(data)
    
    ## Initialise figure
    fig, axs = plt.subplots(1, 3,figsize=(14,4))
    
    ## Plot spectral energy transfer
    axs[0].plot(high_res.ispec_k.data,high_res.ispec_energy_transfer.data,label=r"$256^2$",lw=3,color="black")
    axs[0].plot(low_res.ispec_k.data,low_res.ispec_energy_transfer.data,label=r"$64^2$",linestyle="-.",color="black")
    axs[0].plot(spectral_k,np.mean(reference_spectral[0],axis=0),color="green",label=r"$64^2+\mathcal{L}_\theta$")
    axs[0].fill_between(spectral_k,np.mean(reference_spectral[0],axis=0)-np.std(reference_spectral[0],axis=0),np.mean(reference_spectral[0],axis=0)+np.std(reference_spectral[0],axis=0),color="green",alpha=0.2)
    axs[0].plot(spectral_k,np.mean(spectral_data[0],axis=0),color="red",label=r"$64^2+$%s" % label)
    axs[0].fill_between(spectral_k,np.mean(spectral_data[0],axis=0)-np.std(spectral_data[0],axis=0),np.mean(spectral_data[0],axis=0)+np.std(spectral_data[0],axis=0),color="red",alpha=0.2)
    axs[0].set_xscale("log")
    axs[0].set_xlim(4e-6,2e-4)
    axs[0].set_ylabel(r"$\partial_t E$ [$m^3/s^3$]")
    axs[0].set_xlabel(r"k [$m^{-1}$]")
    axs[0].legend()

    ## Set titles for KE(time) plot
    axs[1].set_title(r"$\mathcal{L}_\theta$")
    axs[2].set_title(r"%s" % label)

    ## Plot KE accumulation
    cols=["blue","green","orange","pink"]
    alphas=[1,5,10,15]
    for aa in range(len(alpha_data)):
        axs[1].plot(model_time,np.mean(reference_alpha[aa],axis=0),color=cols[aa],label=r"$64^2, \alpha=$ %d" % alphas[aa])
        axs[1].fill_between(model_time,np.mean(reference_alpha[aa],axis=0)-np.std(reference_alpha[aa],axis=0),np.mean(reference_alpha[aa],axis=0)+np.std(reference_alpha[aa],axis=0),color=cols[aa],alpha=0.2)
        axs[2].plot(model_time,np.mean(alpha_data[aa],axis=0),color=cols[aa],label=r"$64^2, \alpha=$ %d" % alphas[aa])
        axs[2].fill_between(model_time,np.mean(alpha_data[aa],axis=0)-np.std(alpha_data[aa],axis=0),np.mean(alpha_data[aa],axis=0)+np.std(alpha_data[aa],axis=0),color=cols[aa],alpha=0.2)
    axs[1].legend()

    axs[1].plot(model_time,np.mean(hires,axis=0),color="black",label=r"$256^2, \alpha=1$")
    axs[1].fill_between(model_time,np.mean(hires,axis=0)-np.std(hires,axis=0),np.mean(hires,axis=0)+np.std(hires,axis=0),color="black",alpha=0.2)

    axs[2].plot(model_time,np.mean(hires,axis=0),color="black",label=r"$256^2, \alpha=1$")
    axs[2].fill_between(model_time,np.mean(hires,axis=0)-np.std(hires,axis=0),np.mean(hires,axis=0)+np.std(hires,axis=0),color="black",alpha=0.2)

    ## Some final figure config
    axs[1].set_ylim(0,6000)
    axs[2].set_ylim(0,6000)
    axs[1].set_ylabel("KE (time)")
    axs[1].set_xlabel("Model time (years)")
    axs[2].set_xlabel("Model time (years)")
    axs[2].yaxis.set_ticklabels([])

    plt.subplots_adjust(wspace=0.0,hspace=0.01)
    plt.tight_layout()
    
    return

