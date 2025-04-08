# copied from https://github.com/DENG-MIT/KAN-ODEs/blob/main/Lotka-Volterra-Pytorch/efficient_kan/efficientkan.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from torchdiffeq import odeint_adjoint as torchodeint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import gc
import torch.nn as nn
import sys
from ODE_KAN import KAN
from matplotlib.lines import Line2D



#Generate LV predator-prey data
#dx/dt=alpha*x-beta*x*y
#dy/dt=delta*x*y-gamma*y



def gen_data_pred_prey(X0, alpha, beta, delta, gamma, tf, N_t):
    def pred_prey_deriv(X, t, alpha, beta, delta, gamma):
        x=X[0]
        y=X[1]
        dxdt = alpha*x-beta*x*y
        dydt = delta*x*y-gamma*y
        dXdt=[dxdt, dydt]
        return dXdt
    
    t=np.linspace(0, tf, N_t)

    soln_arr=scipy.integrate.odeint(pred_prey_deriv, X0, t, args=(alpha, beta, delta, gamma))
    print(soln_arr.shape)
    return soln_arr, t


def get_data_lorenz63():
    states = np.loadtxt("Results/results_lorenz/truth.dat")
    t = states[:, 0]
    soln_arr = states[:, 1:]
    for i in range(3):
        soln_arr[:,i] = (soln_arr[:,i]-soln_arr[:,i].mean())/soln_arr[:,i].std()
    return soln_arr, t


class ODEDataset(Dataset):
    def __init__(self, data, times, batch_length, overlap=True):
        self.data = data
        # self.times = times
        self.batch_length = batch_length
        if overlap:
            indexes1 = np.arange(0, len(data), batch_length)
            indexes2 = np.arange(batch_length//2, len(data), len(data)-1)
            self.start_indexes = np.concatenate((indexes1, indexes2), axis=0)
        else:
            self.start_indexes = np.arange(0, len(data), batch_length) 
    
    def __len__(self):
        return len(self.start_indexes)
    
    def __getitem__(self, idx):
        start_index = self.start_indexes[idx]
        end_index = min(start_index + self.batch_length, len(self.data))
        init_cond = self.data[start_index]
        # time_interval = self.times[start_index:end_index]
        solution = self.data[start_index:end_index]
        if end_index - start_index < self.batch_length and start_index-end_index>0:
            raise AttributeError(f"batch length has to divide total length, diff is {end_index-start_index} ")
            # time_interval = torch.cat((time_interval, time_interval.new_full((self.batch_length - len(time_interval),), time_interval[-1].item())))
            # solution = torch.cat((solution, solution.new_full((self.batch_length - len(solution), solution.shape[1]), solution[-1, 0].unsqueeze(0).item())))
        return init_cond, solution 
    
    


class Trainer:
    def __init__(self, n_dims, n_hidden=10, grid_size=5, init_cond=np.array([1,1]), data=None, plot_F=100, tf=14, tf_train=3.5,
                 lr = 2e-3, t=None, model_path="", checkpoint_folder="", checkpoint_freq=100, image_folder=""):
        self.plot_freq = plot_F
        self.cp_freq = checkpoint_freq
        self.tf = tf  # time frame to be used (from zero to this time) 
        self.tf_train = tf_train  # time frame to be used for training (the first part)
        self.samples = data.shape[0]
        self.samples_train = int(tf_train*self.samples/tf)
        
        
        self.plot_lims = [(data[:, i].min(), data[:, i].max()) for i in range(data.shape[1])]
    
         
        
        self.lr = lr     
        self.model = KAN(layers_hidden=[n_dims,n_hidden,n_dims], grid_size=grid_size) #k is order of piecewise polynomial
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_list_train=[]
        self.loss_list_test=[]
        self.loss_min = 1e10
        self.start_epoch = 0
        self.epoch_list_test = []
        if model_path != "":
            self.load_checkpoint(model_path)
    
        self.init_cond = torch.unsqueeze((torch.Tensor(np.transpose(init_cond))), 0)
        self.init_cond.requires_grad=True
        self.soln_arr=torch.tensor(data)
        self.soln_arr.requires_grad=True
        self.soln_arr_train=self.soln_arr[:self.samples_train, :]
        self.t=torch.tensor(t)
        self.t_train =torch.tensor(np.linspace(0, tf_train, self.samples_train))
        print(self.t_train.shape)
        self.cpf = checkpoint_folder
        self.im_f = image_folder
        plt.rc('text', usetex=True)  # use latex for prettier plots
        self.lines = [Line2D([0], [0], color='r', linestyle='dashed', lw=0.5), 
                 Line2D([0], [0], color='black', lw=0.5)]
    
    def load_checkpoint(self, mp):
        checkpoint = torch.load(mp, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_min = checkpoint["loss_min"]
        self.loss_list_test = checkpoint["loss_list_test"]
        self.loss_list_train = checkpoint["loss_list_train"]
        self.start_epoch = checkpoint["start_epoch"]
        self.epoch_list_test = checkpoint["epoch_list_test"]
         
    def save_checkpoint(self,epoch, path):
        torch.save(
            {
            "start_epoch": epoch+1, 
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_min": self.loss_min,
            "loss_list_test": self.loss_list_test,
            "loss_list_train": self.loss_list_train,
            "epoch_list_test": self.epoch_list_test
            },
            path,  
        )
        
    def plotter(self, pred, epoch, t, optimal=False):
    #callback plotter during training, plots current solution
        fig, axarr = plt.subplots(self.soln_arr.shape[1], 1, sharex=True)
        labels=["x", "y", "z"]    # maximum three labels
        for i,ax in enumerate(axarr):
            ax.plot(self.t, self.soln_arr[:, i].detach(), color='black', label="_nolegend_", lw=0.5)
            ax.plot(t, pred[:, i].detach(), linestyle="dashed", color="red", label="_nolegend_", lw=0.5)
            ax.set_ylabel(labels[i])
            ax.axvline(self.tf_train)
            ax.set_ylim(self.plot_lims[i])
        
        axarr[0].legend(self.lines, ['Pred', "Truth"], loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=2)
        # plt.subplots_adjust(top=0.85)
        ax.set_xlabel("time [s]")
        if optimal:
            fig.suptitle(f"Current Optimal is {epoch} epochs")
            fig.savefig(os.path.join(self.im_f, "optimal_pred.pdf"), dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
        else:
            fig.savefig(os.path.join(self.im_f, "training_updates/train_epoch_"+str(epoch)+".pdf"), dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
            plt.close('all')
            plt.figure()
            plt.semilogy(torch.Tensor(self.loss_list_train), label='train')
            plt.semilogy(self.epoch_list_test, torch.Tensor(self.loss_list_test), label='test')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig(os.path.join(self.im_f, "loss.pdf"), dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
        plt.close('all')  
           
    def train(self, num_epochs=10000, val_freq=10, batch_size=16, batch_length=100, n_min=2, n_max=10):
        def calDeriv(t, X):
            dXdt=self.model(X)
            return dXdt
        last = -1
        pred_test=None
        p1 = self.model.layers[0].spline_weight
        p2 = self.model.layers[0].base_weight
        p3 = self.model.layers[1].spline_weight
        p4 = self.model.layers[1].base_weight
        print(self.soln_arr_train.shape)
        n = max(n_min, 2)
        batch_size=1 + int(self.samples_train/(n*4))
        ds = ODEDataset(self.soln_arr_train, self.t_train, n)
        dL = DataLoader(ds, batch_size=batch_size, shuffle=True)
        t_int = self.t_train[:n].float()
        scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,1., 0.01, num_epochs)
        t = self.t
        # train_weights = torch.exp(-alpha * t_int).unsqueeze(1).unsqueeze(2) # add a deecaying importance
        # print(train_weights.shape)
        
        
        print(f"batch_size{batch_size}")
        print(f"batch length {n}")
        
        for epoch in (bar := tqdm(range(self.start_epoch, num_epochs))):
            count = 0
            loss = 0
            t_try = int(n_min + (n_max-n_min)*(epoch/num_epochs)**1.5)
            if t_try > n and self.samples_train % t_try == 0:
                n = t_try
                print(f"batch length {n}")
                batch_size=1 + int(self.samples_train/(n*2))
                print(f"batch_size{batch_size}")
                ds = ODEDataset(self.soln_arr_train, self.t_train, n)
                dL = DataLoader(ds, batch_size=batch_size, shuffle=True)
                t_int = self.t_train[:n]
            for init_cond, sol in dL:
                
                init_cond = init_cond.float()
                # print(init_cond.detach())
                sol = sol.float()
                # print(t_int.detach())
                # print(sol.detach())
                # raise RuntimeError 
                # raise RuntimeError(" Stop here ")
                pred=torchodeint(calDeriv, init_cond, t_int, adjoint_params=[p1, p2, p3, p4], rtol=1e-5, atol=1e-7, method='dopri5')
                
                # predp = pred.detach().numpy()
                # solp = sol.detach().numpy()
                # predp = np.reshape(predp, (18, 2)) 
                # solp = np.reshape(solp, (,2))
                # plt.figure()
                # for i in range(batch_size):
                #     for j in range(2):
                #         plt.plot(t_int, predp[:, i, j], label="pred")
                #         plt.plot(t_int, solp[i,:, j], label="sol")
                # plt.legend()
                # plt.show()
                # print((pred[:, 0, :]-sol).size())
                # print((pred.shape))
                # print((sol.shape))
                # print(pred.detach())
                # print(self.soln_arr_train[i_start:i_start+batch_size, :].shape)
                
                diff=(pred-sol.transpose(0,1))
                loss_train = torch.mean(torch.square(diff))
                
                # print(f"loss: {loss_train.detach().item()}")
                
                # print(loss_train)
                # loss_train.retain_grad()
                loss_train.backward()
                self.optimizer.step()
                # print(init_cond.shape[0])
                loss = (loss*count + init_cond.shape[0]*loss_train.detach().item())/(count+init_cond.shape[0])
                # print(f"mean: {loss}") 
                count += init_cond.shape[0]
            self.loss_list_train.append(loss)
            if epoch % val_freq ==0 or epoch == self.start_epoch:  # always evaluate the first epoch to avoid crashing..
                with torch.no_grad():
                    try: 
                        pred_test=torchodeint(calDeriv, self.init_cond, self.t, adjoint_params=[], rtol=1e-5, atol=1e-7)
                        self.loss_list_test.append(torch.mean(torch.square(pred_test[self.samples_train:,0, :]-self.soln_arr[self.samples_train:, :])).detach().cpu())
                        self.epoch_list_test.append(epoch)
                        t = self.t
                    except AssertionError as e:
                        print(f"Numerical instability encountered at epoch {epoch}: {e}")
                        self.loss_list_test.append(float('inf'))
                        self.epoch_list_test.append(epoch)
                        pred_test = torchodeint(calDeriv, self.init_cond, self.t_train, adjoint_params=[], rtol=1e-5, atol=1e-7)
                        t = self.t_train
            #if epoch ==5:  # seems like they never update the grid....
            #    model.update_grid_from_samples(X0)
            if loss_train<self.loss_min:
                self.loss_min = loss_train
                if self.cp_freq + last <= epoch:
                    self.save_checkpoint(epoch, os.path.join(self.cpf, "best.pt"))
                    self.plotter(pred_test[:,0,:], epoch, t, True)
                    last = int(epoch)
            
            bar.set_postfix(loss=loss, lr=scheduler.get_last_lr())
            if epoch % self.plot_freq ==0:
                self.plotter(pred_test[:,0,:], epoch, t, False)
            scheduler.step()
            if epoch % self.cp_freq == 0:
                self.save_checkpoint(epoch, os.path.join(self.cpf, f"checkpoint_{epoch}.pt"))
        self.save_checkpoint(epoch, os.path.join(self.cpf, "last.pt"))
        self.plotter(pred_test[:,0,:], epoch, t, False)
    
    
    def test_model(self, rollout_lengths=[100]):
        def calDeriv(t, X):
            dXdt=self.model(X)
            return dXdt
        last = -1
        colormap = plt.get_cmap('hsv', len(rollout_lengths)+1)
        colors = [colormap(i/len(rollout_lengths)) for i in range(len(rollout_lengths)+1)]
        fig, axarr = plt.subplots(self.soln_arr.shape[1], 1, sharex=True)
        with torch.no_grad():
            for j,batch_length in enumerate(rollout_lengths):
                print(batch_length)
                ds = ODEDataset(self.soln_arr, self.t, batch_length, False)
                dL = DataLoader(ds, 1, False)
                t = self.t[:batch_length]
                prediction = None
                for init_cond,_ in dL:
                    init_cond = init_cond.float()
                    pred = torchodeint(calDeriv, init_cond, t, adjoint_params=[], method='dopri5')
                    pred = pred[:,0,:].detach().numpy()
                    if prediction is None:
                        prediction = pred
                    else:
                        prediction = np.concatenate((prediction, pred), axis=0)
                for i in range(self.soln_arr.shape[1]):
                    axarr[i].plot(self.t.numpy(), prediction[:,i], linestyle='dashed', color=colors[j], label=f"_nolabel_")
        for i in range(self.soln_arr.shape[1]):
            axarr[i].plot(self.t.numpy(), self.soln_arr[:,i].detach().numpy(), color=colors[-1], label="_nolabel_")
            axarr[i].set_ylim(self.plot_lims[i])
        
        
        
        lines = [Line2D([0], [0], linestyle="dashed", color=colors[i]) for i in range(len(rollout_lengths))] 
        labels = [f"RL={rl}" for rl in rollout_lengths]
        labels.append("Truth")
        lines.append(Line2D([0], [0], color=colors[-1]))
        axarr[0].legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=2)
        plt.tight_layout()
        plt.show()
        


if __name__=='__main__':
    # Train for predator prey
    
    
    
    init_cond = np.array([1,1])
    params = [1.5, 1, 1, 3]
    tf = 14
    N_t = 140
    soln_array, t = gen_data_pred_prey(X0=init_cond, alpha=params[0], beta=params[1],
                                       delta=params[2], gamma=params[3], tf=tf, N_t=N_t)
    trainer = Trainer(n_dims=2, n_hidden=10, grid_size=5, init_cond=init_cond,
                      data=soln_array, t=t, plot_F=100, checkpoint_folder="TrainedModels/ODEKans/LV",
                      tf=tf, tf_train=3.6, lr=0.001, image_folder="images/pred_prey", 
                      checkpoint_freq=200)   #, model_path="TrainedModels/ODEKans/LV/checkpoint_800.pt")
    
    trainer.train(1000, 10, 3, 7, n_min=2, n_max=20)
    
    
    # Train for lorenz 
    
    
    
    # soln_array, t = get_data_lorenz63()
    # # soln_array = soln_array[2500:,:]
    # # t = t[:-2500]
    # init_cond = soln_array[0, :] 
    # trainer = Trainer(n_dims=3, n_hidden=15, grid_size=5, init_cond=init_cond, 
    #                   data=soln_array, t=t, plot_F=100,
    #                   checkpoint_folder="TrainedModels/ODEKans/Lorenz", 
    #                   tf=20, tf_train=2.5, lr=0.001, 
    #                   image_folder="images/Lorenz", model_path="TrainedModels/ODEKans/Lorenz/checkpoint_6000.pt")
    # trainer.train(num_epochs=10000, val_freq=100, batch_size=5, batch_length=100, n_min=25, n_max=100)
    
    
    # Test different rollout lengths
    # soln_array, t = get_data_lorenz63()
    # # soln_array = soln_array[2500:,:]
    # # t = t[:-2500]
    # init_cond = soln_array[0, :] 
    # trainer = Trainer(n_dims=3, n_hidden=15, grid_size=5, init_cond=init_cond, 
    #                   data=soln_array, t=t, plot_F=100,
    #                   checkpoint_folder="TrainedModels/ODEKans/Lorenz", 
    #                   tf=20, tf_train=2.5, lr=0.001, 
    #                   image_folder="images/Lorenz", model_path="TrainedModels/ODEKans/Lorenz/checkpoint_2500.pt")
    # trainer.test_model([100, 500, 2000])
    
    
    
        
            