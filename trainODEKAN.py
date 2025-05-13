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


def plot_train_cycles(models, dims, labels, colors, name_save, name_title):
    plt.figure()
    soln_array, t = get_data_lorenz63(test=False, name="")
    
    init_cond = soln_array[0, :] 
    
    for model, dim, label, color in zip(models, dims, labels, colors):
        tr = Trainer(n_dims=3, n_hidden=dim, grid_size=5, init_cond=init_cond, 
                      data=soln_array, t=t, plot_F=100,
                      checkpoint_folder="TrainedModels/ODEKans/Lorenz", 
                      tf=20, tf_train=2.5, lr=0.001, 
                      image_folder="images/Lorenz", model_path="TrainedModels/ODEKans/Lorenz/"+model, normalize=True)
        plt.semilogy(tr.loss_list_train, label=label, color=color, linewidth=0.5, alpha=.8)
        plt.semilogy(tr.epoch_list_valid, tr.loss_list_valid, label=label+" Valid", color=color, linewidth=0.5, alpha=.8, linestyle='--')
    plt.grid(which="both")
    plt.legend(loc="best", ncols=2, fontsize=14)
    plt.title("Training loss "+ name_title, fontsize=16)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.savefig("images/Lorenz/optimal/LOSS_ODEKAN_"+name_save+".pdf", dpi=200, bbox_inches="tight")
    plt.show() 


def get_data_lorenz63(test=False, name="test.dat"):
    states = np.loadtxt("Results/results_lorenz/truth.dat")
    t = states[:, 0]
    soln_arr = states[:, 1:]
    if test:
        test_states = np.loadtxt("Results/results_lorenz/"+name)
        soln_arr_test = test_states[:, 1:]
        for i in range(3):
            soln_arr_test[:,i] = (soln_arr_test[:,i]-soln_arr[:,i].mean())/soln_arr[:,i].std()
        return soln_arr_test, t
        
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
            indexes2 = np.arange(batch_length//2, len(data)-batch_length, batch_length)
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
    def __init__(self, n_dims, n_hidden=10, grid_size=5, init_cond=np.array([1,1]), data=None, valid=None, plot_F=100, tf=14, tf_train=3.5,
                 lr = 2e-3, t=None, model_path="", checkpoint_folder="", checkpoint_freq=100, image_folder="", normalize=False):
        self.plot_freq = plot_F
        self.cp_freq = checkpoint_freq
        self.tf = tf  # time frame to be used (from zero to this time) 
        self.tf_train = tf_train  # time frame to be used for training (the first part)
        self.samples = data.shape[0]
        self.samples_train = int(tf_train*self.samples/tf)
        
        
        self.plot_lims = [(data[:, i].min(), data[:, i].max()) for i in range(data.shape[1])]
    
        
        self.lr = lr     
        self.model = KAN(layers_hidden=[n_dims,n_hidden,n_dims], grid_size=grid_size, normalize=normalize) #k is order of piecewise polynomial
        p1 = self.model.layers[0].spline_weight
        p2 = self.model.layers[0].base_weight
        p3 = self.model.layers[1].spline_weight
        p4 = self.model.layers[1].base_weight
        p5 = self.model.normalizations[0].bias 
        p6 = self.model.normalizations[0].weight
        self.parameters = [p1, p2, p3, p4, p5, p6]
        self.optimizer = torch.optim.Adam(self.parameters, lr)
        self.loss_list_train=[]
        self.loss_list_valid=[]
        self.loss_min = 1e10
        self.start_epoch = 0
        self.epoch_list_valid = []
        if model_path != "":
            self.load_checkpoint(model_path)
        
        self.valid=None
        if valid is not None:
            self.valid = torch.tensor(valid)
        
            
    
        self.init_cond = torch.unsqueeze((torch.Tensor(np.transpose(init_cond))), 0)
        self.init_cond.requires_grad=False
        self.soln_arr=torch.tensor(data)
        self.soln_arr.requires_grad=False
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
        self.loss_list_valid = checkpoint["loss_list_test"]
        self.loss_list_train = checkpoint["loss_list_train"]
        self.start_epoch = checkpoint["start_epoch"]-1
        self.epoch_list_valid = checkpoint["epoch_list_test"]
        
         
    def save_checkpoint(self,epoch, path):
        torch.save(
            {
            "start_epoch": epoch+1, 
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_min": self.loss_min,
            "loss_list_test": self.loss_list_valid,
            "loss_list_train": self.loss_list_train,
            "epoch_list_test": self.epoch_list_valid
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
            plt.grid(True, which='both')
            plt.semilogy(self.epoch_list_valid, torch.Tensor(self.loss_list_valid), label='valid')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig(os.path.join(self.im_f, "loss.pdf"), dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
            plt.legend()
        plt.close('all')  
           
    def train(self, num_epochs=10000, val_freq=10, batch_size=16, n_min=2, n_max=10, val_len=10):
        def calDeriv(t, X):
            dXdt=self.model(X)
            return dXdt
        last = -1
        pred_test=None
       
        # self.optimizer = torch.optim.Adam(params=[p1, p2, p3, p4], lr=0.001)
        print(self.soln_arr_train.shape)
        n = max(n_min, 2)
        ds = ODEDataset(self.soln_arr, self.t_train, n)
        print(len(ds))
        dL = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        t_int = self.t_train[:n]
        scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,1., 0.1, num_epochs)
        if self.valid is not None:
            ds_valid = ODEDataset(self.valid, self.t_train, val_len)
            dL_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=True, drop_last=False)
            t_int_valid= self.t[:val_len]
        # t = self.t
        # train_weights = torch.exp(-alpha * t_int).unsqueeze(1).unsqueeze(2) # add a deecaying importance
        # print(train_weights.shape)
        
        # self.model(self.soln_arr.float(), update_grid=True)
        print(f"batch_size{batch_size}")
        print(f"batch length {n}")
        for name, param in self.model.named_parameters():
            if 'spline_weight' in name:
                print(f"{name}: weight norm = {param.data.norm().item():.4e}")
        
        for epoch in (bar := tqdm(range(self.start_epoch, num_epochs))):
            count = 0
            loss = 0
            sum_loss = 0
            if epoch % 1000 == 0:
                a = int(round(n_min + (n_max-n_min)*(epoch/num_epochs)**1.5))
                n = a if self.samples % a == 0 else n
                print(f"RL={n}, batch size ={batch_size}")
                ds = ODEDataset(self.soln_arr, self.t_train, n)
                print(len(ds))
                dL = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
                t_int = self.t_train[:n]

            self.model.train()
            for init_cond, sol in dL:
                init_cond = init_cond.float()
                sol = sol.float()
                pred=torchodeint(calDeriv, init_cond, t_int, adjoint_params=self.parameters, rtol=1e-5, atol=1e-7, method='dopri5') 
                diff=torch.square((pred-sol.transpose(0,1))[1:,:,:])   # remove zero prediction
                loss_train = torch.mean(diff)
                sum_loss +=diff.sum().detach().item()
                # loss_reg = self.model.regularization_loss(regularize_activation=0, regularize_entropy=0.0001)
                # loss_train += loss_reg  # might help with the generalization
                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()
                
                count += diff.numel()
            loss = sum_loss/count
            self.loss_list_train.append(loss)
            # if epoch==10:
            #     break
            self.model.eval()
            
            if epoch % val_freq ==0 or epoch == self.start_epoch:  # always evaluate the first epoch to avoid crashing..
                for name, parameter in self.model.named_parameters():
                    print(name)
                    print(parameter.grad)
                    print(parameter.requires_grad)
                    print(parameter.data)
                if self.valid is not None:
                    for init_cond, sol in dL_valid:
                        init_cond = init_cond.float()
                        sol = sol.float()
                        pred=torchodeint(calDeriv, init_cond, t_int_valid, adjoint_params=[], rtol=1e-5, atol=1e-7, method='dopri5') 
                        diff=torch.square((pred-sol.transpose(0,1)))
                        sum_loss +=diff.sum().detach().item()
                        count += diff.numel()
                    self.loss_list_valid.append(sum_loss/count)
                    self.epoch_list_valid.append(epoch)
                with torch.no_grad():
                    try: 
                        pred_test=torchodeint(calDeriv, self.init_cond, self.t, adjoint_params=[], rtol=1e-5, atol=1e-7)
                        t=self.t
                    except AssertionError as e:
                        print(f"Numerical instability encountered at epoch {epoch}: {e}")
                        pred_test = torch.zeros((self.soln_arr.shape[0], 1, self.soln_arr.shape[1]))
                        t = self.t
            # if epoch % 10==0 and epoch != 0:
            #     self.model(self.soln_arr.float(), update_grid=True)
            #if epoch ==5:  # seems like they never update the grid....
            #    model.update_grid_from_samples(X0)
            if self.valid is not None:
                if self.loss_list_valid[-1]<self.loss_min:
                    self.loss_min = self.loss_list_valid[-1]
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
    
    
    def test_model(self, rollout_lengths=[100], model_name="MODEL", title="Title"):
        plt.rc('lines', linewidth=0.5)
        self.model.eval()
        def calDeriv(t, X):
            dXdt=self.model(X)
            return dXdt
        last = -1
        colormap = plt.get_cmap('hsv', len(rollout_lengths)+1)
        colors = [colormap(i/len(rollout_lengths)) for i in range(len(rollout_lengths)+1)]
        fig, axarr = plt.subplots(self.soln_arr.shape[1], 1, sharex=True)
        with torch.no_grad():
            for j,batch_length in enumerate(rollout_lengths):
                # print(batch_length)
                ds = ODEDataset(self.soln_arr, self.t, batch_length, False)
                dL = DataLoader(ds, 1, False)
                t = self.t[:batch_length]
                prediction = None
                for init_cond, sol in dL:
                    init_cond = init_cond.float()
                    pred = torchodeint(calDeriv, init_cond, t, adjoint_params=[], method='dopri5')
                    pred = pred[:,0,:].detach().numpy()
                    zpe= sol[0, :, :].detach().numpy()-init_cond.detach().numpy()
                    if prediction is None:
                        prediction = pred
                        zero_pred_error = zpe
                    else:
                        prediction = np.concatenate((prediction, pred), axis=0)
                        # print(zpe.shape)
                        zero_pred_error = np.concatenate((zero_pred_error, zpe), axis=0)
                    
                for i in range(self.soln_arr.shape[1]):
                    axarr[i].plot(self.t.numpy(), prediction[:,i], linestyle='dashed', color=colors[j], label=f"_nolabel_")
                    print(f"mse channgel {i}, RL={batch_length} is given by {np.square(prediction[:,i]-self.soln_arr[:,i].detach().numpy()).mean()}")
                    print(f"comparisson to zero gradient channel {i}, RL={batch_length} is {np.square(zero_pred_error[:, i]).mean()} ")
        ylabs = ["x", "y", "z"]
        for i in range(self.soln_arr.shape[1]):
            sarr = self.soln_arr[:,i].detach().numpy()
            axarr[i].plot(self.t.numpy(), sarr, color="black", label="_nolabel_")
            axarr[i].set_ylim(self.plot_lims[i])
            axarr[i].set_ylabel(ylabs[i])
        
        fig.supxlabel("time [s]")
        
        
        
        lines = [Line2D([0], [0], linestyle="dashed", color=colors[i]) for i in range(len(rollout_lengths))] 
        labels = [f"RL={rl}" for rl in rollout_lengths]
        labels.append("Truth")
        lines.append(Line2D([0], [0], color="black"))
        axarr[0].legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=4)
        fig.suptitle("Test loss. " + title)
        plt.tight_layout()
        fig.savefig("images/Lorenz/optimal/"+model_name+"test_error.pdf", dpi=200, bbox_inches="tight")
        plt.show()
    
    


def plot_errors_lorenz(model, data, filepath, title, savename):
    model.eval()
    traindata = np.loadtxt("Results/results_lorenz/truth.dat")[:, 1:]
    test = np.loadtxt(filepath)[:, 1:]
    def lorenz63_deriv(state):
        beta = 2.6666666666666665
        rho = 28
        sigma=10
        dx = (sigma * (state[:,1] - state[:,0]))/traindata[:,0].std()
        dy = (rho*state[:,0] - state[:,1] - state[:,0]*state[:,2])/traindata[:,1].std()
        dz = (state[:,0]*state[:,1] - beta*state[:,2])/traindata[:, 2].std()
        return np.array([dx, dy, dz]).T
    deriv = lorenz63_deriv(test)
    norm_deriv = np.linalg.norm(deriv, axis=1)
    errors = np.linalg.norm((model(data.float()).detach().numpy()-deriv), axis=1)/norm_deriv
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=errors, cmap="RdBu_r", vmin=0, vmax=np.percentile(errors, 99.5))
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Relative error', fontsize=16)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("Gradient error along trajectory. " + title, fontsize=18)
    plt.tight_layout()
    plt.savefig("images/Lorenz/optimal/error_trajectory_"+savename +".pdf", dpi=200, bbox_inches='tight')
    plt.show()
    plt.figure()
    bins = np.logspace(np.log10(errors.min()), np.log10(errors.max()), 20)
    plt.hist(errors, bins=bins)
    plt.title("Gradient error. " + title,)
    plt.xlabel("Relative error", fontsize=14)
    plt.ylabel(r"\#samples", fontsize=14)
    plt.xscale('log')
    plt.savefig("images/Lorenz/optimal/hist_errors_"+savename+".pdf", dpi=200, bbox_inches='tight')
    plt.show()
    
    
def plot_errors_fieldwise_lorenz(model, data, filepath, title, savename):
    model.eval()
    traindata = np.loadtxt("Results/results_lorenz/truth.dat")[:, 1:]
    test = np.loadtxt(filepath)[:, 1:]
    def lorenz63_deriv(state):
        beta = 2.6666666666666665
        rho = 28
        sigma=10
        dx = (sigma * (state[:,1] - state[:,0]))/traindata[:,0].std()
        dy = (rho*state[:,0] - state[:,1] - state[:,0]*state[:,2])/traindata[:,1].std()
        dz = (state[:,0]*state[:,1] - beta*state[:,2])/traindata[:, 2].std()
        return np.array([dx, dy, dz]).T
    deriv = lorenz63_deriv(test)
    # norm_deriv = np.linalg.norm(deriv, axis=1)
    errors = np.abs(model(data.float()).detach().numpy()-deriv)/np.abs(deriv)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=errors, cmap="RdBu_r", vmin=0, vmax=np.percentile(errors, 99.5))
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Relative error', fontsize=16)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("Gradient error along trajectory. " + title, fontsize=18)
    plt.tight_layout()
    plt.savefig("images/Lorenz/optimal/error_trajectory_"+savename +".pdf", dpi=200, bbox_inches='tight')
    plt.show()
    plt.figure()
    bins = np.logspace(np.log10(errors.min()), np.log10(errors.max()), 20)
    plt.hist(errors, bins=bins)
    plt.title("Gradient error. " + title,)
    plt.xlabel("Relative error", fontsize=14)
    plt.ylabel(r"\#samples", fontsize=14)
    plt.xscale('log')
    plt.savefig("images/Lorenz/optimal/hist_errors_"+savename+".pdf", dpi=200, bbox_inches='tight')
    plt.show()
       
        


if __name__=='__main__':
    # # Train for predator prey
    
    
    
    # init_cond = np.array([1,1])
    # params = [1.5, 1, 1, 3]
    # tf = 14
    # N_t = 140
    # soln_array, t = gen_data_pred_prey(X0=init_cond, alpha=params[0], beta=params[1],
    #                                    delta=params[2], gamma=params[3], tf=tf, N_t=N_t)
    # trainer = Trainer(n_dims=2, n_hidden=10, grid_size=5, init_cond=init_cond,
    #                   data=soln_array, t=t, plot_F=100, checkpoint_folder="TrainedModels/ODEKans/LV",
    #                   tf=tf, tf_train=3.6, lr=0.001, image_folder="images/pred_prey", 
    #                   checkpoint_freq=200)   #, model_path="TrainedModels/ODEKans/LV/checkpoint_800.pt")
    
    # trainer.train(1000, 10, 3, 7, n_min=2, n_max=20)
    
    
    # Train for lorenz 
    
    
    
    # soln_array, t = get_data_lorenz63()
    # valid, _ = get_data_lorenz63(test=True, name="valid.dat")
    # init_cond = soln_array[0, :] 
    # trainer = Trainer(n_dims=3, n_hidden=6, grid_size=5, init_cond=init_cond, 
    #                   data=soln_array, t=t, plot_F=200, checkpoint_freq=500, 
    #                   checkpoint_folder="TrainedModels/ODEKans/Lorenz/1step_train", 
    #                   tf=20, tf_train=16, lr=0.001, valid=valid,
    #                   image_folder="images/Lorenz", normalize=True, model_path="TrainedModels/ODEKans/Lorenz/1step_train/checkpoint_6000.pt")
    # trainer.train(num_epochs=10000, val_freq=200, batch_size=128, n_min=2, n_max=6)
    
    
    # Test different rollout lengths
    # soln_array, t = get_data_lorenz63(test=True, name="test.dat")
    # init_cond = soln_array[0, :] 
    # trainer = Trainer(n_dims=3, n_hidden=4, grid_size=5, init_cond=init_cond, data = soln_array, t=t, plot_F=200, checkpoint_freq=500,
    #                   checkpoint_folder="TrainedModels/ODEKans/Lorenz", 
    #                   tf=20, tf_train=2.5, lr=0.01, 
    #                   image_folder="images/Lorenz", model_path="TrainedModels/ODEKans/Lorenz/1step_train/MODEL4NODES_IR_Final.pt", normalize=True)
    # trainer.test_model([50, 100, 500], model_name="4hidden_Ir", title="4 hidden IR")

    
    # models = ["1step_train/MODEL6NODES.pt", "1step_train/MODEL6NODES_wreg.pt", "MODEL6NODES_IR_nureg.pt", "MODEL_6nodes_IR.pt"]
    # models = ["1step_train/FINALCR6_FINAL.pt", "1step_train/last.pt"]
    # models = ["1step_train/MODEL4NODES_CR_Final.pt", "1step_train/MODEL4NODES_IR_FINAL.pt"]
    # # dims = [6, 4, 6, 4]
    # dims = [4, 4]
    # # labels = ["6 hidden CR", "4 hidden CR", "6 hidden IR", "4 hidden IR"]
    # labels = ["CR", "IR"]
    # colors = ["r", "k"]
    # plot_train_cycles(models, dims, labels, colors, name_save="4nodes", name_title="4 Nodes")
    
    
    
    soln_array, t = get_data_lorenz63(test=True, name="test.dat")
    init_cond = soln_array[0, :] 
    trainer = Trainer(n_dims=3, n_hidden=6, grid_size=5, init_cond=init_cond, data = soln_array, t=t, plot_F=200, checkpoint_freq=500,
                      checkpoint_folder="TrainedModels/ODEKans/Lorenz", 
                      tf=20, tf_train=2.5, lr=0.01, 
                      image_folder="images/Lorenz", model_path="TrainedModels/ODEKans/Lorenz/1step_train/last.pt", normalize=True)
    
    plot_errors_lorenz(trainer.model, trainer.soln_arr, "Results/results_lorenz/test.dat", "6 nodes IR", savename="6nodesIR")
    
    
        
            