import matplotlib.animation as animation
from IPython.display import HTML
import torch
import numpy as np
import pyqg_explorer.util.transforms as transforms


class DiffusionAnimation():
    def __init__(self,sim,model,fps=10,nSteps=1000):
        """ Takes a torchqg sim, a diffusion model, and animates the forward diffusion
            process. Will plot the noise level and MSE between true and noised fields as
            the diffusion process evolves """
        self.sim=sim
        self.model=model
        self.fps = fps
        self.nFrames = nSteps
        self.mse=np.array([])
        self.criterion=torch.nn.MSELoss()
        self.timesx=np.linspace(0,1000,1001)
        self.i=0

        x_upper = transforms.normalise_field(self.sim.q[0],config["q_mean_upper"],config["q_std_upper"])
        x_lower = transforms.normalise_field(self.sim.q[1],config["q_mean_lower"],config["q_std_lower"])
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
        axs[1][3].set_ylim(3e-5,1e0)
        axs[1][3].set_xlim(0,self.model.timesteps)

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

    def __animate(self):
        fig, axs = plt.subplots(2, 4,figsize=(12,5))
        self.ax1=axs[0][0].imshow(self.q_ds[0].data[0], cmap=cmocean.cm.balance)
        fig.colorbar(self.ax1, ax=axs[0][0])
        axs[0][0].set_xticks([]); axs[0][0].set_yticks([])
        axs[0,0].set_title("Before noise")

        self.ax2=axs[0][1].imshow(self.q_ds[0].data[0], cmap=cmocean.cm.balance, interpolation='none')
        fig.colorbar(self.ax2, ax=axs[0][1])
        axs[0][1].set_xticks([]); axs[0][1].set_yticks([])
        axs[0,1].set_title("noise added")

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

