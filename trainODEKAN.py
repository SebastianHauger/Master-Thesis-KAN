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
    return soln_arr, t


class ODEDataset(Dataset):
    def __init__(self, data, times, batch_length):
        self.data = data
        # self.times = times
        self.batch_length = batch_length
        self.start_indexes = np.arange(0, len(data), batch_length)
    
    def __len__(self):
        return len(self.start_indexes)
    
    def __getitem__(self, idx):
        start_index = self.start_indexes[idx]
        end_index = min(start_index + self.batch_length, len(self.data))
        init_cond = self.data[start_index]
        # time_interval = self.times[start_index:end_index]
        solution = self.data[start_index:end_index]
        if start_index - end_index < self.batch_length:
            # time_interval = torch.cat((time_interval, time_interval.new_full((self.batch_length - len(time_interval),), time_interval[-1].item())))
            solution = torch.cat((solution, solution.new_full((self.batch_length - len(solution), solution.shape[1]), solution[-1, 0].unsqueeze(0).item())))
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
        
    def plotter(self, pred, epoch, optimal=False):
    #callback plotter during training, plots current solution
        fig, axarr = plt.subplots(self.soln_arr.shape[1], 1, sharex=True)
        labels=["x", "y", "z"]    # maximum three labels
        for i,ax in enumerate(axarr):
            ax.plot(self.t, self.soln_arr[:, i].detach(), color='black', label="Truth", lw=0.5)
            ax.plot(self.t, pred[:, i].detach(), linestyle="dashed", color="red", label="Prediction", lw=0.5)
            ax.set_ylabel(labels[i])
            ax.axvline(self.tf_train)
            ax.set_ylim(self.plot_lims[i])
        
        ax.set_xlabel("time [s]")
        fig.legend(loc="center right")
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
           
    def train(self, num_epochs=10000, val_freq=10, batch_size=16, batch_length=100 ):
        def calDeriv(t, X):
            dXdt=self.model(X)
            return dXdt
        last = -1
        pred_test=None
        p1 = self.model.layers[0].spline_weight
        p2 = self.model.layers[0].base_weight
        p3 = self.model.layers[1].spline_weight
        p4 = self.model.layers[1].base_weight
        ds = ODEDataset(self.soln_arr_train, self.t_train, batch_length)
        dL = DataLoader(ds, batch_size=batch_size, shuffle=True)
        t_int = self.t_train[:batch_length].float()
        
        
        for epoch in (bar := tqdm(range(self.start_epoch, num_epochs))):
            count = 0
            loss = 0
            for init_cond, sol in dL:
                # print(init_cond.size())
                # print(t_int.size())
                # print(sol.size())
                init_cond = init_cond.float()
                sol = sol.float()
                
                # raise RuntimeError(" Stop here ")
                pred=torchodeint(calDeriv, init_cond, t_int, adjoint_params=[p1, p2, p3, p4])
                # print(pred[:,0,:].shape)
                
                # print(self.soln_arr_train[i_start:i_start+batch_size, :].shape)
                loss_train=torch.mean(torch.square(pred[:, 0, :]-sol))
                
                # print(loss_train)
                loss_train.retain_grad()
                loss_train.backward()
                self.optimizer.step()
                loss = (loss*count + loss_train.detach().item())/(count+1)
                count += 1
            self.loss_list_train.append(loss_train.detach().cpu())
            if epoch % val_freq ==0 or epoch == self.start_epoch:  # always evaluate the first epoch to avoid crashing..
                with torch.no_grad():
                    pred_test=torchodeint(calDeriv, self.init_cond, self.t, adjoint_params=[])
                    self.loss_list_test.append(torch.mean(torch.square(pred_test[self.samples_train:,0, :]-self.soln_arr[self.samples_train:, :])).detach().cpu())
                    self.epoch_list_test.append(epoch)
            #if epoch ==5:  # seems like they never update the grid....
            #    model.update_grid_from_samples(X0)
            if loss_train<self.loss_min:
                self.loss_min = loss_train
                if self.cp_freq + last <= epoch:
                    self.save_checkpoint(epoch, os.path.join(self.cpf, "best.pt"))
                    self.plotter(pred_test[:,0,:], epoch, True)
                    last = int(epoch)
            
            bar.set_postfix(loss=loss_train.detach().item())
            if epoch % self.plot_freq ==0:
                self.plotter(pred_test[:,0,:], epoch, False)
            
            if epoch % self.cp_freq == 0:
                self.save_checkpoint(epoch, os.path.join(self.cpf, f"checkpoint_{epoch}.pt"))
        self.save_checkpoint(epoch, os.path.join(self.cpf, "last.pt"))
        self.plotter(pred_test[:,0,:], epoch, False)
        


if __name__=='__main__':
    # tf=14
    # tf_train=3.5
    # N_t_train=35
    # N_t=int((35*tf/tf_train))
    # lr=2e-3
    # num_epochs=10000
    # plot_freq=100
    # alpha=1.5
    # beta=1
    # gamma=3
    # delta=1
    # X0 = np.array([1,1])
    # soln_array, t = gen_data_pred_prey(X0, alpha, beta, delta, gamma, tf, N_t)
    # trainer = Trainer(X0, soln_array, tf=tf, tf_train=tf_train,
    #                   samples_train=N_t_train, lr=lr, t=t, checkpoint_freq=200, plot_F=200,
    #                   model_path="TrainedModels/ODEKans/last.pt", checkpoint_folder="TrainedModels/ODEKans")
    # trainer.train(num_epochs=2000, val_freq=10)
    
    soln_array, t = get_data_lorenz63()
    init_cond = soln_array[0, :]
    
    trainer = Trainer(n_dims=3, n_hidden=10, grid_size=5, init_cond=init_cond, 
                      data=soln_array, t=t, plot_F=10,
                      checkpoint_folder="TrainedModels/ODEKans/Lorenz", 
                      tf=100, tf_train=40, lr=0.1, 
                      image_folder="images/Lorenz", model_path="TrainedModels/ODEKans/Lorenz/last.pt")
    trainer.train(num_epochs=600, val_freq=2, batch_size=10)
    
    
    
        
            