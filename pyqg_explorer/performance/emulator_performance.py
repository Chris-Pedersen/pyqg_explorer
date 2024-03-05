import numpy as np
import torch
import torch.nn as nn
import pyqg_explorer.util.transforms as transforms
import matplotlib.pyplot as plt
import xarray as xr
import cmocean
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.stats import pearsonr


class EmulatorMSE():
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
            
        self.simlist_pred=[]
        self.simlist_true=[]
        
    @torch.no_grad
    def _get_next_step(self,q_i):
        """ For a given field at time i, use the attributed model to push the system forward
            to time i+dt (dt is stored as the time horizon in model config) """

        q_i_dt=self.network(q_i.unsqueeze(0))
        q_i_dt=q_i_dt.squeeze()

        return q_i_dt+q_i
    
    def get_short_MSEs(self,return_data=False):
        ds=xr.load_dataset("/scratch/cp3759/pyqg_data/sims/emulator_trajectory_sims/torch_%s_%sk.nc" % (self.eddy,self.network.config["increment"]))
        times=np.arange(self.network.config["increment"],ds.q.shape[1]*self.network.config["increment"],self.network.config["increment"])
        
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
            for bb in range(1,ds.q.shape[1]):
                q_i=self._get_next_step(q_i)
                q_i_true=self.normalise(torch.tensor(ds.q[aa,bb].values,device=self.device))
                mses[bb-1]=(criterion(q_i_true,q_i))
                mses_0[bb-1]=(criterion(q_i_true,init))
                
            #print(mses.shape)
                
            plt.title("MSE(truth,emulator) in blue, MSE(truth at t=0, truth at t=i) in red")
            plt.plot(times,mses,color="blue",label="Emulator MSE wrt truth",alpha=0.2)
            plt.plot(times,mses_0,color="red",label="True MSE wrt t=0",alpha=0.2)
            plt.yscale("log")
            plt.ylim(1e-3,5e1)
            plt.xlabel("timestep")
            plt.ylabel("MSE")

            #sim=torch_model.PseudoSpectralModel(nx=64,dt=3600,dealias=True,parameterization=torch_param.Smagorinsky())
            #sim.set_q1q2(self.denormalise(q_i))
            #self.simlist_pred.append(sim)
            #sim=torch_model.PseudoSpectralModel(nx=64,dt=3600,dealias=True,parameterization=torch_param.Smagorinsky())
            #sim.set_q1q2(ds.q[aa,bb].values)
            #self.simlist_true.append(sim)
        return fig

    def normalise(self,q):
        ## Map from physical to normalised space using the factors used to train the network
        ## Normalise each field individually, then cat arrays back to shape appropriate for a torch model
        x_upper = transforms.normalise_field(q[0],self.network.config["q_mean_upper"],self.network.config["q_std_upper"])
        x_lower = transforms.normalise_field(q[1],self.network.config["q_mean_lower"],self.network.config["q_std_lower"])
        x = torch.stack((x_upper,x_lower),dim=0)
        return x
    
    def denormalise(self,q):
        ## Map from physical to normalised space using the factors used to train the network
        ## Normalise each field individually, then cat arrays back to shape appropriate for a torch model
        x_upper = transforms.denormalise_field(q[0],self.network.config["q_mean_upper"],self.network.config["q_std_upper"])
        x_lower = transforms.denormalise_field(q[1],self.network.config["q_mean_lower"],self.network.config["q_std_lower"])
        x = torch.stack((x_upper,x_lower),dim=0)
        return x


class DenoiserMSE():
    """ Object to store performance tests relevant to neural emulators """
    def __init__(self,network,denoiser,denoise_timestep=10,denoise_delay=500,denoise_interval=5,path="/scratch/cp3759/pyqg_data/plots/denoiser_plots"):
        """ network:  Torch model we want to test. Assuming this is a model for the subgrid forcing
            valid_loader: validation loader from EmulatorDatasetTorch """
        
        
        if torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"
        
        self.path=path
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
        self.simlist_pred=[]
        self.simlist_true=[]
        self.simlist_dn=[]
        
    @torch.no_grad
    def _get_next_step(self,q_i):
        """ For a given field at time i, use the attributed model to push the system forward
            to time i+dt (dt is stored as the time horizon in model config) """

        q_i_dt=self.network(q_i.unsqueeze(0))
        q_i_dt=q_i_dt.squeeze()

        return q_i_dt+q_i
    
    def test_denoiser(self):
        ds=xr.load_dataset("/scratch/cp3759/pyqg_data/sims/emulator_trajectory_sims/torch_%s_%sk.nc" % (self.eddy,self.network.config["increment"]))
        times=np.arange(self.network.config["increment"],ds.q.shape[1]*self.network.config["increment"],self.network.config["increment"])
        
        criterion=nn.MSELoss()
        fig=plt.figure()
        plt.figure(figsize=(16,5))
        plt.suptitle("Delay=%d, Timestep=%d, Interval=%d" % (self.denoise_delay,self.denoise_timestep,self.denoise_interval))
        ## This can all be parallelised to make use of GPU..
        for aa in range(20):
            mses=np.empty(len(times))
            mses_0=np.empty(len(times))
            mses_dn=np.empty(len(times))
            
            corr=np.empty(len(times))
            corr_0=np.empty(len(times))
            corr_dn=np.empty(len(times))
            
            cos=np.empty(len(times))
            cos_0=np.empty(len(times))
            cos_dn=np.empty(len(times))
            
            q_i=self.normalise(torch.tensor(ds.q[aa,0].to_numpy(),device=self.device,dtype=torch.float32))
            q_i_dn=q_i
            ## Index counter for loss arrays
            cc=0
            init=self.normalise(torch.tensor(ds.q[aa,0].values,device=self.device,dtype=torch.float32))
            
            
            ## Bool to keep track of when to apply denoiser
            should_denoise=False
            for bb in range(1,ds.q.shape[1]):
                q_i_true=self.normalise(torch.tensor(ds.q[aa,bb].values,device=self.device,dtype=torch.float32))
                ## Emulator rollout
                q_i=self._get_next_step(q_i)
                ## Denoised emulator rollout
                q_i_dn=self._get_next_step(q_i_dn)
                
                ## Determine if we want to denoise
                should_denoise = (bb % self.denoise_interval == 0) and (bb>self.denoise_delay)
                if should_denoise:
                    q_i_dn=self.denoiser.denoising(q_i_dn.unsqueeze(0),self.denoise_timestep)
                    q_i_dn=q_i_dn.squeeze()
                
                ## Calcluate MSEs
                mses[bb-1]=(criterion(q_i_true,q_i))
                mses_0[bb-1]=(criterion(q_i_true,init))
                mses_dn[bb-1]=(criterion(q_i_true,q_i_dn))
                
                ## Calculate correlations
                corr[bb-1]=(pearsonr(q_i_true.cpu().numpy().flatten(),q_i.cpu().numpy().flatten())[0])
                corr_0[bb-1]=(pearsonr(q_i_true.cpu().numpy().flatten(),init.cpu().numpy().flatten())[0])
                corr_dn[bb-1]=(pearsonr(q_i_true.cpu().numpy().flatten(),q_i_dn.cpu().numpy().flatten())[0])
                cc+=1
                
            plt.subplot(1,3,1)
            plt.title("MSE")
            plt.plot(times,mses,color="black",label="Emulator MSE wrt truth",alpha=0.2)
            plt.plot(times,mses_0,color="blue",label="True MSE wrt t=0",alpha=0.2)
            plt.plot(times,mses_dn,color="red",label="True MSE wrt t=0",alpha=0.4)
            plt.xlabel("timestep")
            plt.yscale("log")
            plt.ylim(1e-3,2e1)
            plt.xlabel("timestep")
            plt.ylabel("MSE")
            plt.subplot(1,3,2)
            plt.title("Correlation")
            plt.plot(times,corr,color="black",label="Emulator MSE wrt truth",alpha=0.2)
            plt.plot(times,corr_0,color="blue",label="True MSE wrt t=0",alpha=0.2)
            plt.plot(times,corr_dn,color="red",label="True MSE wrt t=0",alpha=0.4)
            plt.xlabel("timestep")
            
            plt.subplot(1,3,3)
            plt.title("Correlation difference")
            plt.plot(times,corr_dn-corr,color="black",label="Emulator MSE wrt truth",alpha=0.2)
            plt.xlabel("timestep")
            
            sim=torch_model.PseudoSpectralModel(nx=64,dt=3600,dealias=True,parameterization=torch_param.Smagorinsky())
            sim.set_q1q2(self.denormalise(q_i))
            self.simlist_pred.append(sim)
            sim=torch_model.PseudoSpectralModel(nx=64,dt=3600,dealias=True,parameterization=torch_param.Smagorinsky())
            sim.set_q1q2(ds.q[aa,bb].values)
            self.simlist_true.append(sim)
            sim=torch_model.PseudoSpectralModel(nx=64,dt=3600,dealias=True,parameterization=torch_param.Smagorinsky())
            sim.set_q1q2((self.denormalise(q_i_dn)))
            self.simlist_dn.append(sim)
            
        plt.savefig(self.path+"/delay%d_timestep%d_interval%d_trajectory.pdf" % (self.denoise_delay,self.denoise_timestep,self.denoise_interval))

        return fig

    def normalise(self,q):
        ## Map from physical to normalised space using the factors used to train the network
        ## Normalise each field individually, then cat arrays back to shape appropriate for a torch model
        x_upper = transforms.normalise_field(q[0],self.network.config["q_mean_upper"],self.network.config["q_std_upper"])
        x_lower = transforms.normalise_field(q[1],self.network.config["q_mean_lower"],self.network.config["q_std_lower"])
        x = torch.stack((x_upper,x_lower),dim=0)
        return x
    
    def denormalise(self,q):
        ## Map from physical to normalised space using the factors used to train the network
        ## Normalise each field individually, then cat arrays back to shape appropriate for a torch model
        x_upper = transforms.denormalise_field(q[0],self.network.config["q_mean_upper"],self.network.config["q_std_upper"])
        x_lower = transforms.denormalise_field(q[1],self.network.config["q_mean_lower"],self.network.config["q_std_lower"])
        x = torch.stack((x_upper,x_lower),dim=0)
        return x
    
    def spectral_diagnostics(self):
        """ Take a true sim, and some denoised sims. Plot spectra. We are just assuming the first
            set of sims are truth, which we'll plot in black, and the second are some kind of comparison:
            noised or denoised, which we'll plot in red """

        fig, axs = plt.subplots(2, 3,figsize=(10,5))


        axs[0,0].set_title("KE spectrum")

        axs[0,1].set_title("Enstrophy spectrum")

        axs[0,2].set_title("q pdf")

        plt.suptitle("Delay=%d, Timestep=%d, Interval=%d" % (self.denoise_delay,self.denoise_timestep,self.denoise_interval))
        ## Set ylimits for eddy spectra
        axs[0,0].set_ylim(1e-3,5e3)
        axs[1,0].set_ylim(1e-3,1e2)
        axs[0,1].set_ylim(5e-10,6e-5)
        axs[1,1].set_ylim(8e-11,1e-6)

        for aa,sim in enumerate(self.simlist_true):
            kes=sim.get_KE_ispec()
            ens=sim.get_enstrophy_ispec()

            ## Kinetic energy spectra
            axs[0,0].loglog(sim.k1d_plot,kes[0],color="black",alpha=0.3)
            axs[1,0].loglog(sim.k1d_plot,kes[1],color="black",alpha=0.3)

            ## Enstrophy spectra
            axs[0,1].loglog(sim.k1d_plot,ens[0],color="black",alpha=0.3)
            axs[1,1].loglog(sim.k1d_plot,ens[1],color="black",alpha=0.3)

            ux,uy=util.PDF_histogram(sim.q[0].cpu().numpy().flatten())
            axs[0,2].semilogy(ux,uy,color="black",alpha=0.3)

            vx,vy=util.PDF_histogram(sim.q[1].cpu().numpy().flatten())
            axs[1,2].semilogy(vx,vy,color="black",alpha=0.3)

        for aa,sim in enumerate(self.simlist_pred):
            kes=sim.get_KE_ispec()
            ens=sim.get_enstrophy_ispec()

            ## Kinetic energy spectra
            axs[0,0].loglog(sim.k1d_plot,kes[0],color="blue",alpha=0.3)
            axs[1,0].loglog(sim.k1d_plot,kes[1],color="blue",alpha=0.3)

            ## Enstrophy spectra
            axs[0,1].loglog(sim.k1d_plot,ens[0],color="blue",alpha=0.3)
            axs[1,1].loglog(sim.k1d_plot,ens[1],color="blue",alpha=0.3)

            ux,uy=util.PDF_histogram(sim.q[0].cpu().numpy().flatten())
            axs[0,2].semilogy(ux,uy,color="blue",alpha=0.3)

            vx,vy=util.PDF_histogram(sim.q[1].cpu().numpy().flatten())
            axs[1,2].semilogy(vx,vy,color="blue",alpha=0.3)

        for aa,sim in enumerate(self.simlist_dn):
            kes=sim.get_KE_ispec()
            ens=sim.get_enstrophy_ispec()

            ## Kinetic energy spectra
            axs[0,0].loglog(sim.k1d_plot,kes[0],color="red",alpha=0.3)
            axs[1,0].loglog(sim.k1d_plot,kes[1],color="red",alpha=0.3)

            ## Enstrophy spectra
            axs[0,1].loglog(sim.k1d_plot,ens[0],color="red",alpha=0.3)
            axs[1,1].loglog(sim.k1d_plot,ens[1],color="red",alpha=0.3)

            ux,uy=util.PDF_histogram(sim.q[0].cpu().numpy().flatten())
            axs[0,2].semilogy(ux,uy,color="red",alpha=0.3)

            vx,vy=util.PDF_histogram(sim.q[1].cpu().numpy().flatten())
            axs[1,2].semilogy(vx,vy,color="red",alpha=0.3)
            
        plt.savefig(self.path+"/delay%d_timestep%d_interval%d_spectra.pdf" % (self.denoise_delay,self.denoise_timestep,self.denoise_interval))

        return fig

class EmulatorAnimation():
    def __init__(self,q_ds,model,fps=10,nSteps=1000,normalise=True):
        self.q_ds=q_ds
        self.model=model
        self.fps = fps
        self.nSteps = nSteps
        self.nFrames = int(self.nSteps/self.model.config["increment"])
        self.q_i_pred=q_ds[0].data
        self.normalise=normalise
        self.mse=[]
        self.correlation_upper=[]
        self.correlation_lower=[]
        self.autocorrelation_upper=[]
        self.autocorrelation_lower=[]
        self.criterion=nn.MSELoss()
        self.times=np.arange(0,self.nFrames*self.model.config["increment"]+0.01,self.model.config["increment"])
        
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
        axs[0][0].set_title("Simulation")

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
        self.ds_i=i*self.model.config["increment"]
        self.time_text.set_text("%d timesteps" % (i*self.model.config["increment"]))
    
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