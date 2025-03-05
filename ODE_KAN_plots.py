# copied from https://github.com/DENG-MIT/KAN-ODEs/blob/main/Lotka-Volterra-Pytorch/efficient_kan/efficientkan.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from torchdiffeq import odeint_adjoint as torchodeint
from tqdm import tqdm
import os
import gc
import torch.nn as nn
import sys
from ODE_KAN import KAN



#Generate LV predator-prey data
#dx/dt=alpha*x-beta*x*y
#dy/dt=delta*x*y-gamma*y



def gen_data_pred_prey(x0, y0, alpha, beta, delta, gamma, tf, N_t):
    def pred_prey_deriv(X, t, alpha, beta, delta, gamma):
        x=X[0]
        y=X[1]
        dxdt = alpha*x-beta*x*y
        dydt = delta*x*y-gamma*y
        dXdt=[dxdt, dydt]
        return dXdt

    X0=np.array([x0, y0])
    t=np.linspace(0, tf, N_t)

    soln_arr=scipy.integrate.odeint(pred_prey_deriv, X0, t, args=(alpha, beta, delta, gamma))
    return X0, t, soln_arr


def gen_data_lorenz63():
    pass


class Trainer:
    def __init__(self, init_cond=np.array([1,1]), data=None, plot_F=100, tf=14, tf_train=3.5, samples_train=35,
                 lr = 2e-3, t=None, save_freq=1000, model_path="", checkpoint_path=""):
        self.plot_freq = plot_F
        self.tf = tf  # time frame to be used (from zero to this time) 
        self.tf_train = tf_train  # time frame to be used for training (the first part)
        self.samples_train = samples_train  # decides the frequency of samples 
        self.samples = int((samples_train*tf/tf_train))
        self.lr = lr
        self.x0 = init_cond[0]
        self.y0 = init_cond[1]        
        self.save_freq = save_freq
        self.model = KAN(layers_hidden=[2,10,2], grid_size=5) #k is order of piecewise polynomial
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_list_train=[]
        self.loss_list_test=[]
        self.min_loss = 1e10
        if checkpoint_path != "":
            self.load_checkpoint()
    
        X0=torch.unsqueeze((torch.Tensor(np.transpose(X0))), 0)
        X0.requires_grad=True
        self.soln_arr=torch.Tensor(data)
        self.soln_arr.requires_grad=True
        self.soln_arr_train=self.soln_arr[:samples_train, :]
        self.t=torch.Tensor(t)
        self.t_train =torch.tensor(np.linspace(0, tf_train, samples_train))
        self.cpp = checkpoint_path
    
    
    def load_checkpoint(self, mp):
        checkpoint = torch.load(mp, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.min_loss = checkpoint["min_loss"]
        self.loss_list_test = checkpoint["loss_list_test"]
        self.loss_list_train = checkpoint["loss_list_train"]
        self.start_epoch = checkpoint["start_epoch"]
        

    
    def save_checkpoint(self,epoch, path):
        torch.save(
            {
            "start_epoch": epoch+1, 
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "min_loss": self.min_loss,
            "loss_list_test": self.loss_list_test,
            "loss_list_train": self.loss_list_train
            },
            path,  
        )
        
    def plotter(self, pred, epoch, optimal=False):
    #callback plotter during training, plots current solution
        plt.figure()
        plt.plot(self.t, self.soln_arr[:, 0].detach(), color='g')
        plt.plot(self.t, self.soln_arr[:, 1].detach(), color='b')
        plt.plot(self.t, pred[:, 0].detach(), linestyle='dashed', color='g')
        plt.plot(self.t, pred[:, 1].detach(), linestyle='dashed', color='b')

        plt.legend(['x_data', 'y_data', 'x_KAN-ODE', 'y_KAN-ODE'])
        plt.ylabel('concentration')
        plt.xlabel('time')
        plt.ylim([0, 8])
        plt.vlines(self.tf_train, 0, 8)
        if optimal:
            plt.title(f"Current Optimal is {epoch} epochs")
            plt.savefig("images/pred_prey/optimal_pred.png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
        else:
            plt.savefig("images/pred_prey/training_updates/train_epoch_"+str(epoch) +".png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
            plt.close('all')
            plt.figure()
            plt.semilogy(torch.Tensor(self.loss_list_train), label='train')
            plt.semilogy(torch.Tensor(self.loss_list_test), label='test')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig("images/pred_prey/loss.png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
        plt.close('all')
        
        
    def train(self, num_epochs=10000, val_freq=10):
        def calDeriv(t, X):
            dXdt=self.model(X)
            return dXdt
        opt_plot_counter=0

        epoch_cutoff=10 #start at smaller lr to initialize, then bump it up

        p1 = self.model.layers[0].spline_weight
        p2 = self.model.layers[0].base_weight
        p3 = self.model.layers[1].spline_weight
        p4 = self.model.layers[1].base_weight
        for epoch in (bar := tqdm(range(num_epochs))):
            opt_plot_counter+=1
            #if epoch==epoch_cutoffs[2]:
            #    model = kan.KAN(width=[2,3,2], grid=grids[1], k=3).initialize_from_another_model(model, X0_train)
            self.optimizer.zero_grad()

            pred=torchodeint(calDeriv, self.X0, self.t_train, adjoint_params=[p1, p2, p3, p4])
            loss_train=torch.mean(torch.square(pred[:, 0, :]-self.soln_arr_train))
            loss_train.retain_grad()
            loss_train.backward()
            self.optimizer.step()
            self.loss_list_train.append(loss_train.detach().cpu())
            if epoch % val_freq ==0:
                with torch.no_grad():
                    pred_test=torchodeint(calDeriv, self.init_cond, self.t, adjoint_params=[])
                    self.loss_list_test.append(torch.mean(torch.square(pred_test[self.samples_train:,0, :]-self.soln_arr[self.samples_train:, :])).detach().cpu())
            #if epoch ==5:  # seems like they never update the grid....
            #    model.update_grid_from_samples(X0)
            if loss_train<self.loss_min:
                self.loss_min = loss_train
                if opt_plot_counter>=200:
                    print('plotting optimal model')
                    self.save_checkpoint(epoch, os.path.join(self.cpf, "best.pt"))
                    self.plotter(pred_test[:,0,:], epoch, True)
                    opt_plot_counter=0
            
            bar.set_postfix(loss=loss_train.detach().item())
            # print('Iter {:04d} | Train Loss {:.5f}'.format(epoch, loss_train.item()))
            ##########
            #########################make a checker that deepcopys the best loss into, like, model_optimal
            #########
            ######################and then save that one into the file, not just whatever the current one is
            if epoch % self.plot_freq ==0:
                #model.save_ckpt('ckpt_predprey')
                self.plotter(pred_test[:,0,:], epoch, False)
            
            if epoch % self.cp_freq == 0:
                self.save_checkpoint(epoch, os.path.join(self.cpf, f"checkpoint_{epoch}.pt"))
        self.save_checkpoint(epoch, os.path.join(self.cpf, "last.pt"))
        


if __name__=='__main__':
    pass
    
            