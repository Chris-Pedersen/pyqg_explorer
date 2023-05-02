import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import xarray as xr

import pyqg_explorer.util.powerspec as powerspec
import pyqg_explorer.parameterizations.parameterizations as parameterizations
import pyqg_explorer.generate_datasets as generate_datasets
import cmocean


class EmulatorPerformance():
    """ Object to store performance tests relevant to neural emulators """
    def __init__(self):
        return


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
        self.y_true=[]
        self.y_pred=[]
        
        count=0
        ## Cache x, true y and predicted y values that we will use to guage offline performance
        for data in valid_loader:
            x=data[0]
            y=data[1]
            count+=x.shape[0]
            self.x_np.append(x)
            self.y_true.append(y)
            self.y_pred.append(self.network(x[:,0:2,:]))
            if count>threshold:
                break
        self.x_np=torch.vstack(self.x_np).detach().numpy()
        self.y_true=torch.vstack(self.y_true).detach().numpy()
        self.y_pred=torch.vstack(self.y_pred).detach().numpy() ## Do we really need to save this?
        
        ## Save R2, since we basically get it for free at this point
        self.r2_upper=r2_score(self.x_np[:,2,:,:].flatten(),self.y_pred[:,0,:,:].flatten())
        self.r2_lower=r2_score(self.x_np[:,3,:,:].flatten(),self.y_pred[:,1,:,:].flatten())
    
    ## Offline tests
    def get_power_spectrum(self):
        """ Plot power spectra of true and predicted subgrid forcing """
        power_upper_true=[]
        power_lower_true=[]
        power_upper_pred=[]
        power_lower_pred=[]

        for aa in range(len(self.x_np)):
            power_upper_true.append(powerspec.get_power_spectrum(self.x_np[aa][2]))
            power_lower_true.append(powerspec.get_power_spectrum(self.x_np[aa][3]))
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
        ax=axs[0].hist2d(self.x_np[:,2,:,:].flatten(),self.y_pred[:,0,:,:].flatten(),bins=100,range=[[-4,4],[-4,4]],cmap='RdPu');
        fig.colorbar(ax[3], ax=axs[0])
        axs[1].plot(line,line,linestyle="dashed",color="gray",alpha=0.5)
        ax=axs[1].hist2d(self.x_np[:,3,:,:].flatten(),self.y_pred[:,1,:,:].flatten(),bins=100,range=[[-4,4],[-4,4]],cmap='RdPu');
        fig.colorbar(ax[3], ax=axs[1])
        return fig
        
    def get_fields(self,map_index=None):
        
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

        image=self.x_np[map_index][2]
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

        image=self.x_np[map_index][2]-self.y_pred[map_index][0]
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

        image=self.x_np[map_index][3]
        limit=np.max(np.abs(image))
        ax=axs[1][1].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[1][1])
        axs[1][1].set_xticks([]); axs[1][1].set_yticks([])

        image=self.y_pred[map_index][1]
        limit=np.max(np.abs(image))
        ax=axs[1][2].imshow(image, cmap=cmocean.cm.balance,vmin=-limit,vmax=limit, interpolation='none')
        fig.colorbar(ax, ax=axs[1][2])
        axs[1][2].set_xticks([]); axs[1][2].set_yticks([])

        image=self.x_np[map_index][3]-self.y_pred[map_index][1]
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

