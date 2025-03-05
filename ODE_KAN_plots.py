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



def gen_data(x0, y0, alpha, beta, delta, gamma, tf, N_t):
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



    




    







class Trainer:
    def __init__(self, num_epochs, plot_F, tf=14, tf_train=3.5, samples_train=35,
                 lr = 2e-3, init_cond=[1,1], save_freq=1000, model_name=""):
        self.num_epochs = num_epochs
        self.plot_freq = plot_F
        self.tf = tf  # time frame to be used (from zero to this time) 
        self.tf_train = tf_train  # time frame to be used for training (the first part)
        self.samples_train = samples_train  # decides the frequency of samples 
        self.samples = int((samples_train*tf/tf_train))
        self.lr = lr
        self.x0 = init_cond[0]
        self.y0 = init_cond[1]
        alpha=1.5
        beta=1
        gamma=3
        delta=1
        X0, t, soln_arr = gen_data(x0, y0, alpha, beta, delta, gamma)
        ##coefficients from https://arxiv.org/pdf/2012.07244
        
        self.save_freq = save_freq
        model = KAN(layers_hidden=[2,10,2], grid_size=5) #k is order of piecewise polynomial
        if model_name != "":
            model.load_state_dict("Trained models/"+model_name, )
            return model
    
        X0=torch.unsqueeze((torch.Tensor(np.transpose(X0))), 0)
        X0.requires_grad=True
        soln_arr=torch.Tensor(soln_arr)
        soln_arr.requires_grad=True
        soln_arr_train=soln_arr[:N_t_train, :]
        t=torch.Tensor(t)
        t_learn=torch.tensor(np.linspace(0, tf_learn, N_t_train))
        def calDeriv(t, X):
            dXdt=model(X)
            return dXdt
        loss_list_train=[]
        loss_list_test=[]
        #initialize ADAM optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)



        if is_restart==True:
            model.load_ckpt('ckpt_predprey')

        loss_min=1e10 #arbitrarily large to overwrite later
        opt_plot_counter=0

        epoch_cutoff=10 #start at smaller lr to initialize, then bump it up

        p1=model.layers[0].spline_weight
        p2=model.layers[0].base_weight
        p3=model.layers[1].spline_weight
        p4=model.layers[1].base_weight
        for epoch in (bar := tqdm(range(num_epochs))):
            opt_plot_counter+=1
            #if epoch==epoch_cutoffs[2]:
            #    model = kan.KAN(width=[2,3,2], grid=grids[1], k=3).initialize_from_another_model(model, X0_train)
            optimizer.zero_grad()

            pred=torchodeint(calDeriv, X0, t_learn, adjoint_params=[p1, p2, p3, p4])
            loss_train=torch.mean(torch.square(pred[:, 0, :]-soln_arr_train))
            loss_train.retain_grad()
            loss_train.backward()
            optimizer.step()
            loss_list_train.append(loss_train.detach().cpu())
            pred_test=torchodeint(calDeriv, X0, t, adjoint_params=[])
            loss_list_test.append(torch.mean(torch.square(pred_test[N_t_train:,0, :]-soln_arr[N_t_train:, :])).detach().cpu())
            #if epoch ==5:
            #    model.update_grid_from_samples(X0)
            if loss_train<loss_min:
                loss_min=loss_train
                #model.save_ckpt('ckpt_predprey_opt')
                if opt_plot_counter>=200:
                    print('plotting optimal model')
                    plotter_opt( pred_test[:, 0, :], soln_arr, epoch, loss_list_train, loss_list_test, t)
                    opt_plot_counter=0
            
            bar.set_postfix(loss=loss_train.detach().item())
            # print('Iter {:04d} | Train Loss {:.5f}'.format(epoch, loss_train.item()))
            ##########
            #########################make a checker that deepcopys the best loss into, like, model_optimal
            #########
            ######################and then save that one into the file, not just whatever the current one is
            if epoch % plot_freq ==0:
                #model.save_ckpt('ckpt_predprey')
                plotter(pred_test[:, 0, :], soln_arr, epoch, loss_list_train, loss_list_test, t)
            
            if epoch % save_freq == 0:
                torch.save(model.state_dict, "Trained models/ODE.pt")
            
    
    
    def plotter(pred, soln_arr, epoch, loss_train, loss_test, tf_learn):
    #callback plotter during training, plots current solution
        plt.figure()
        plt.plot(t, soln_arr[:, 0].detach(), color='g')
        plt.plot(t, soln_arr[:, 1].detach(), color='b')
        plt.plot(t, pred[:, 0].detach(), linestyle='dashed', color='g')
        plt.plot(t, pred[:, 1].detach(), linestyle='dashed', color='b')

        plt.legend(['x_data', 'y_data', 'x_KAN-ODE', 'y_KAN-ODE'])
        plt.ylabel('concentration')
        plt.xlabel('time')
        plt.ylim([0, 8])
        plt.vlines(tf_learn, 0, 8)
        plt.savefig("images/pred_prey/training_updates/train_epoch_"+str(epoch) +".png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
        plt.close('all')
        
        plt.figure()
        plt.semilogy(torch.Tensor(loss_train), label='train')
        plt.semilogy(torch.Tensor(loss_test), label='test')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig("images/pred_prey/loss.png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
        plt.close('all')
    
    def plotter_opt(pred, soln_arr, epoch, loss_train, loss_test, t):
        #plots the optimal solution 
        plt.figure()
        plt.plot(t, soln_arr[:, 0].detach(), color='g')
        plt.plot(t, soln_arr[:, 1].detach(), color='b')
        plt.plot(t, pred[:, 0].detach(), linestyle='dashed', color='g')
        plt.plot(t, pred[:, 1].detach(), linestyle='dashed', color='b')

        plt.legend(['x_data', 'y_data', 'x_KAN-ODE', 'y_KAN-ODE'])
        plt.ylabel('concentration')
        plt.xlabel('time')
        plt.ylim([0, 8])
        plt.vlines(tf_learn, 0, 8)
        plt.savefig("images/pred_prey/optimal/train_trial_.png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
        plt.close('all')

        plt.close('all')
        


if __name__=='__main__':
    pass
    
            