import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch
import numpy as np
import pyqg_explorer.util.transforms as transforms
import torch_qg.util as util
import torch_qg.model as torch_model
import torch_qg.parameterizations as torch_param
import cmocean


class DiffusionAnimation():
    """ Animation of the forward diffusion process for a 2 layer system """
    def __init__(self,sim,model,fps=10,nSteps=1000,savestring=None):
        self.sim=sim
        self.model=model
        self.fps = fps
        self.nFrames = nSteps
        self.savestring = savestring
        self.mse=np.array([])
        self.criterion=torch.nn.MSELoss()
        self.timesx=np.linspace(0,1000,1001)
        self.i=0
        
        x_upper = transforms.normalise_field(self.sim.q[0],self.model.config["q_mean_upper"],self.model.config["q_std_upper"])
        x_lower = transforms.normalise_field(self.sim.q[1],self.model.config["q_mean_lower"],self.model.config["q_std_lower"])
        self.q_orig = torch.stack((x_upper,x_lower),dim=0)
        self.q_to_noise=self.q_orig.unsqueeze(0)
        self.q_orig=self.q_orig.cpu().numpy()
        self._push_forward()
        
        
    def noise_image(self,q_to_noise,t):
        t=torch.tensor([t],dtype=torch.int64).to(device)
        noise=torch.randn_like(q_to_noise).to(denoise_sim.device)
        return noise,model.sqrt_alphas_cumprod.gather(-1,t).reshape(q_to_noise.shape[0],1,1,1)*q_to_noise+model.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(q_to_noise.shape[0],1,1,1)*noise
    
        
    def _push_forward(self):
        """ Update variable quantities - the noise field, residual, timestep, MSE """
        self.noise,self.noised=self.noise_image(self.q_to_noise,self.i)
        mse=self.criterion(self.q_to_noise,self.noised)
        self.mse=np.append(self.mse,mse.cpu().numpy())
        self.noise=self.noise.squeeze().cpu().numpy()
        self.noised=self.noised.squeeze().cpu().numpy()
        
        return
    
    def animate(self):
        fig, axs = plt.subplots(2, 4,figsize=(12,5))
        axs[0,0].set_title("Before noise")
        self.ax1=axs[0,0].imshow(self.q_orig[0],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax1, ax=axs[0][0])
        self.ax2=axs[1,0].imshow(self.q_orig[1],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax2, ax=axs[1][0])

        axs[0,1].set_title("noise added")
        self.ax3=axs[0,1].imshow(self.noise[0],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax3, ax=axs[0][1])
        self.ax4=axs[1,1].imshow(self.noise[1],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax4, ax=axs[1][1])

        axs[0,2].set_title("noised fields")
        self.ax5=axs[0,2].imshow(self.noised[0],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax5, ax=axs[0][2])
        self.ax6=axs[1,2].imshow(self.noised[1],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax6, ax=axs[1][2])

        axs[0,3].set_title("Noise level + MSEloss")
        #self.ax7=axs[0,3].plot(1-(model.alphas_cumprod[:self.t].cpu()))
        self.ax7=[axs[0][3].plot(-1),axs[0][3].plot(-1)]
        axs[0,3].set_ylim(0,1)
        axs[0,3].set_xlim(0,1000)

        self.ax8=[axs[1][3].plot(1),axs[1][3].plot(1)]
        axs[1][3].set_yscale("log")
        axs[1][3].set_ylim(3e-5,1e1)
        axs[1][3].set_xlim(0,self.model.timesteps)
        
        fig.tight_layout()
        
        anim = animation.FuncAnimation(
                                       fig, 
                                       self.animate_func, 
                                       frames = self.nFrames,
                                       interval = 1000 / self.fps, # in ms
                                       )
        plt.close()
        
        if self.savestring:
            print("saving")
            # saving to m4 using ffmpeg writer 
            writervideo = animation.FFMpegWriter(fps=self.fps) 
            anim.save('%s.mp4' % self.savestring, writer=writervideo) 
            plt.close()
        else:
            return HTML(anim.to_html5_video())
        
        
    def animate_func(self,i):
        if i % self.fps == 0:
            print( '.', end ='' )
            
        self.i=i
    
        ## Set image and colorbar for each panel
        image=self.q_orig[0]
        self.ax1.set_array(image)
        self.ax1.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.q_orig[1]
        self.ax2.set_array(image)
        
        self.ax2.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.noise[0]
        self.ax3.set_array(image)
        self.ax3.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.noise[1]
        self.ax4.set_array(image)
        self.ax4.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.noised[0]
        self.ax5.set_array(image)
        self.ax5.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.noised[1]
        self.ax6.set_array(image)
        self.ax6.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
 
        self.ax7[0][0].set_xdata(np.array(self.timesx[0:i]))
        self.ax7[0][0].set_ydata(np.array(1-self.model.alphas_cumprod.cpu()[:i]))
        
        self.ax8[0][0].set_xdata(np.array(self.timesx[0:i]))
        self.ax8[0][0].set_ydata(np.array(self.mse[:i]))

        self._push_forward()
        
        return

class ReverseDiffusionAnimation():
    """ Animation of the reverse diffusion process (denoising example) for a 2 layer system """
    def __init__(self,q_input,model,fps=10,nSteps=1000,savestring=None):
        self.q_input=torch.tensor(q_input,dtype=torch.float32)
        self.model=model
        self.fps = fps
        self.nFrames = nSteps
        self.savestring = savestring
        self.mse_u=np.array([])
        self.mse_l=np.array([])
        self.criterion=torch.nn.MSELoss()
        self.timesx=np.linspace(0,nSteps,nSteps+1)
        self.i=nSteps
        
        x_upper = transforms.normalise_field(self.q_input[0],self.model.config["q_mean_upper"],self.model.config["q_std_upper"])
        x_lower = transforms.normalise_field(self.q_input[1],self.model.config["q_mean_lower"],self.model.config["q_std_lower"])
        self.q_orig = torch.stack((x_upper,x_lower),dim=0).unsqueeze(0)
        
        t=(torch.ones(len(self.q_orig),dtype=torch.int64)*self.nFrames).to(device)
        noise=torch.randn_like(self.q_orig).to(device)
        self.noised=model._forward_diffusion(self.q_orig,t,noise)
        self.q_0=self.q_orig ## For loss evaluations, keep unsqeueezed and on gpu
        self.q_orig=self.q_orig.squeeze().cpu().numpy()
        self.x_t=self.noised
        self.noised=self.noised.squeeze().cpu().numpy()
        
        self._push_backward()
    
        
    def _push_backward(self):
        """ Perform a single denoising step """
        
        noise=torch.randn_like(self.x_t).to(device)
        self.x_t_old=self.x_t.squeeze().cpu().numpy()
        t=torch.tensor([self.i for _ in range(len(self.x_t))]).to(device)
        self.x_t=self.model._reverse_diffusion(self.x_t,t,noise)
        
        mse_u=self.criterion(self.q_0[:,0],self.x_t[:,0])
        mse_l=self.criterion(self.q_0[:,1],self.x_t[:,1])
        
        self.denoised=self.x_t.squeeze().cpu().numpy()
        
        self.mse_u=np.append(self.mse_u,mse_u.cpu().numpy())
        self.mse_l=np.append(self.mse_l,mse_l.cpu().numpy())
        
        
        return
    
    def animate(self):
        fig, axs = plt.subplots(2, 5,figsize=(12,5))
        axs[0,0].set_title("Before noise")
        self.ax1=axs[0,0].imshow(self.q_orig[0],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax1, ax=axs[0][0])
        self.ax2=axs[1,0].imshow(self.q_orig[1],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax2, ax=axs[1][0])

        axs[0,1].set_title("Noised image")
        self.ax3=axs[0,1].imshow(self.noised[0],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax3, ax=axs[0][1])
        self.ax4=axs[1,1].imshow(self.noised[1],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax4, ax=axs[1][1])
        
        axs[0,2].set_title("Noise removed")
        self.ax5=axs[0,2].imshow(self.denoised[0],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax5, ax=axs[0][2])
        self.ax6=axs[1,2].imshow(self.denoised[1],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax6, ax=axs[1][2])

        axs[0,3].set_title("Denoising...")
        self.ax7=axs[0,3].imshow(self.denoised[0],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax5, ax=axs[0][2])
        self.ax8=axs[1,3].imshow(self.denoised[1],cmap=cmocean.cm.balance)
        #fig.colorbar(self.ax6, ax=axs[1][2])

        axs[0,4].set_title("Reconstruction MSE loss")
        #self.ax7=axs[0,3].plot(1-(model.alphas_cumprod[:self.t].cpu()))
        self.ax9=[axs[0][4].plot(-1),axs[0][4].plot(-1)]
        axs[0,4].set_ylim(0,max(self.mse_u))
        axs[0,4].set_xlim(0,self.nFrames)

        self.ax10=[axs[1][4].plot(1),axs[1][4].plot(1)]
        axs[1][4].set_ylim(0,max(self.mse_l))
        axs[1][4].set_xlim(0,self.nFrames)
        
        fig.tight_layout()
        
        anim = animation.FuncAnimation(
                                       fig, 
                                       self.animate_func, 
                                       frames = self.nFrames,
                                       interval = 1000 / self.fps, # in ms
                                       )
        plt.close()
        
        if self.savestring:
            print("saving")
            # saving to m4 using ffmpeg writer 
            writervideo = animation.FFMpegWriter(fps=self.fps) 
            anim.save('%s.mp4' % self.savestring, writer=writervideo) 
            plt.close()
        else:
            return HTML(anim.to_html5_video())
        
        
    def animate_func(self,i):
        if i % self.fps == 0:
            print( '.', end ='' )
            
        self.i=self.nFrames-i
    
        ## Set image and colorbar for each panel
        image=self.q_orig[0]
        self.ax1.set_array(image)
        self.ax1.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.q_orig[1]
        self.ax2.set_array(image)
        
        self.ax2.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.noised[0]
        self.ax3.set_array(image)
        self.ax3.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.noised[1]
        self.ax4.set_array(image)
        self.ax4.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.denoised[0]-self.x_t_old[0]
        self.ax5.set_array(image)
        self.ax5.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.denoised[1]-self.x_t_old[1]
        self.ax6.set_array(image)
        self.ax6.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.denoised[0]
        self.ax7.set_array(image)
        self.ax7.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.denoised[1]
        self.ax8.set_array(image)
        self.ax8.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
 
        self.ax9[0][0].set_xdata(np.array(self.timesx[0:i]))
        self.ax9[0][0].set_ydata(np.array(self.mse_u[:i]))
        
        self.ax10[0][0].set_xdata(np.array(self.timesx[0:i]))
        self.ax10[0][0].set_ydata(np.array(self.mse_l[:i]))

        self._push_backward()
        
        return


def field_to_sims(valid_imgs,denoised,config):
    """ For tensor of validation images, and denosied images, return lists of sims
        set with potential vorticity in each of these images. The idea is to use
        the torch_qg simulation object to calculate the relevant diagnostics. The
        sim list can then be passed to spectral_diagnostics to plot the spectra """

    if config["eddy"]:
        add_config={}
    else:
        ## If system is jet, add jet config to sim so we get the right
        ## stream function inversion in KE calculation
        add_config=torch_model.jet_config

    clean_sims=[]
    denoised_sims=[]
    for aa in range(len(valid_imgs)):
        x_upper = transforms.denormalise_field(valid_imgs[aa][0],config["q_mean_upper"],config["q_std_upper"])
        x_lower = transforms.denormalise_field(valid_imgs[aa][1],config["q_mean_lower"],config["q_std_lower"])
        q_true=torch.stack((x_upper,x_lower),dim=0)
        true=torch_model.PseudoSpectralModel(nx=64,dt=3600,dealias=True,parameterization=torch_param.Smagorinsky(),**add_config)
        true.set_q1q2(q_true)
        clean_sims.append(true)

        x_upper = transforms.denormalise_field(denoised[aa][0],config["q_mean_upper"],config["q_std_upper"])
        x_lower = transforms.denormalise_field(denoised[aa][1],config["q_mean_lower"],config["q_std_lower"])
        q_dn=torch.stack((x_upper,x_lower),dim=0)
        denoised_sim=torch_model.PseudoSpectralModel(nx=64,dt=3600,dealias=True,parameterization=torch_param.Smagorinsky(),**add_config)
        denoised_sim.set_q1q2(q_dn)
        denoised_sims.append(denoised_sim)
    return clean_sims,denoised_sims


def spectral_diagnostics(sims,sims2,epoch,eddy=True):
    """ Take a true sim, and some denoised sims. Plot spectra. We are just assuming the first
        set of sims are truth, which we'll plot in black, and the second are some kind of comparison:
        noised or denoised, which we'll plot in red """

    fig, axs = plt.subplots(2, 3,figsize=(10,5))


    axs[0,0].set_title("KE spectrum")
    
    axs[0,1].set_title("Enstrophy spectrum")
    
    axs[0,2].set_title("q pdf")
    
    if eddy:
        plt.suptitle("Spectra and distributions at epoch %d, eddy" % epoch)
        ## Set ylimits for eddy spectra
        axs[0,0].set_ylim(1e-3,5e2)
        axs[1,0].set_ylim(1e-3,1e1)
        axs[0,1].set_ylim(5e-10,6e-6)
        axs[1,1].set_ylim(8e-11,1e-7)
    else:
        plt.suptitle("Spectra and distributions at epoch %d, jet" % epoch)
        ## Set ylimits for jet spectra
        axs[0,0].set_ylim(1e-3,5e2)
        axs[1,0].set_ylim(1e-4,1e2)
        axs[0,1].set_ylim(5e-10,6e-6)
        axs[1,1].set_ylim(1e-11,1e-8)
    
    
    for aa,sim in enumerate(sims):
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
        
    for aa,sim in enumerate(sims2):
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

    return fig


def plot_fields(valid_imgs,noised,denoised,noise_loss,denoise_loss,epoch):
    """ For a set of validation images, noised images, denoised images
        plot a random sample. Also show MSE of the noising process, and the
        denoised MSE """

    plt.figure(figsize=(8,4))
    plt.suptitle("Denoised fields at epoch=%d" % epoch)

    valid_idx=np.random.randint(len(valid_imgs))
    fig_denoise=plt.subplot(2,3,1)
    plt.title("Original field")
    plt.imshow(valid_imgs[valid_idx][0].cpu().numpy(),cmap=cmocean.cm.balance)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(valid_imgs[valid_idx][1].cpu().numpy(),cmap=cmocean.cm.balance)
    plt.colorbar()

    plt.subplot(2,3,2)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title("Noised field: %.2f" % noise_loss)
    plt.imshow(noised[valid_idx][0].cpu().numpy(),cmap=cmocean.cm.balance)
    plt.colorbar()
    plt.subplot(2,3,5)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(noised[valid_idx][1].cpu().numpy(),cmap=cmocean.cm.balance)
    plt.colorbar()

    plt.subplot(2,3,3)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title("Denoised field: %.2f" % denoise_loss)
    plt.imshow(denoised[valid_idx][0].cpu().numpy(),cmap=cmocean.cm.balance)
    plt.colorbar()
    plt.subplot(2,3,6)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(denoised[valid_idx][1].cpu().numpy(),cmap=cmocean.cm.balance)
    plt.colorbar()

    plt.tight_layout()

    return fig_denoise
