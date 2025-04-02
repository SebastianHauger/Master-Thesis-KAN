"""root of the simulation. Run files from here."""
from torch.utils.data import DataLoader
from new_UKAN_proposal import UKAN
# from the_well.benchmark.models.unet_classic import UNetClassic
from the_well.data import WellDataset, WellDataModule
from UNET_classic import UNetClassic
from tqdm import tqdm 
from data import get_dataset
from einops import rearrange
import torch
import matplotlib.pyplot as plt
import numpy as np 
import random 
from the_well.benchmark.metrics import VRMSE
import os


PATH_TO_BASE_HOME = "datasets"
PATH_TO_BASE_ALVIS = "/mimer/NOBACKUP/groups/shallow_ukan/datasets"
IMAGE_DIR = "images/SWE"
MODEL_DIR = "TrainedModels/UStructures"

def get_data_helper(split, home):
    if home:
        split="test"
        return get_dataset(split, PATH_TO_BASE_HOME)
    else:
        return get_dataset(split, PATH_TO_BASE_ALVIS)


def train_UKAN(epochs, lr, device, bs=64, home=False, padding='uniform'):
    print(device)
    train = get_data_helper("train", home)
    val = get_data_helper("valid", home)
    train_loader = DataLoader(train, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val, batch_size=bs, shuffle=False)
    # model=UKAN(padding=padding)
    model = UNetClassic(dim_in=3, dim_out=3, dset_metadata=train.metadata, init_features=32)
    model=model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for i in tqdm(range(epochs)):
        for i,batch in (bar := tqdm(enumerate(train_loader))):
            x = batch["input_fields"]
            x = x.to(device)
            x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
            y = batch["output_fields"]
            y = y.to(device)
            y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")
            yp = model(x)
            mse = (yp-y).square().mean()
            mse.backward()
            optimizer.step()
            optimizer.zero_grad()
            bar.set_postfix(loss=mse.detach().item())
            if i == 30:
                break
        # with torch.no_grad():
        #     avg_mse=0
        #     for n,batch in enumerate(val_loader):
        #         x = batch["input_fields"]
        #         x = x.to(device)
        #         x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
        #         y = batch["output_fields"]
        #         y = y.to(device)
        #         y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")
        #         yp = model(x)
                
        #         mse = (yp-y).square().mean()
        #         avg_mse = (avg_mse*n + mse)/(n+1) # mooving average MSE
        #     print(f"validation loss epoch {i}: {avg_mse}")
    torch.save(model.state_dict(), 'UKAN.pth')
    return model 
                
                 
def test_model(model, device, bs=64, home=False):
    test = get_data_helper("test", home)
    test_loader = DataLoader(test, bs, shuffle=False)
    with torch.no_grad():
        vrmse = 0
        for n,batch in tqdm(enumerate(test_loader)):
            x = batch["input_fields"]
            x = x.to(device)
            x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
            y = batch["output_fields"]
            y = y.to(device)
            y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")
            yp = model(x)
            vrmse_batch = torch.sqrt(torch.mean((y-yp)**2))
            vrmse = (vrmse*n + vrmse_batch)/(n+1)
        print(f"vrmse test set: {vrmse}")
            
            
def load_trained_UKAN_ptfile(name,device, KAN):
    """assuming model has been trained with cuda."""
    if KAN:
        model = UKAN(padding="asym_all")
    else:
        dataset = get_dataset("train", PATH_TO_BASE_HOME, True)
        model = UNetClassic(3, 3, dataset.metadata)
    if device == 'cpu':
        checkpoint = torch.load(os.path.join(MODEL_DIR,name), map_location=torch.device('cpu'), weights_only=False)
    else:
        checkpoint = torch.load(os.path.join(MODEL_DIR,name), weights_only=False)
    print(checkpoint.keys())
    print(checkpoint["validation_loss"])
    print(checkpoint["epoch"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model

def load_trained_UKAN_pth_file(name, device):
    model = UKAN(padding="asym_all")
    if device == "cpu":
        checkpoint = torch.load(os.path.join(MODEL_DIR,name), map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(os.path.join(MODEL_DIR,name))
    model.load_state_dict(checkpoint)
    model.to(device)
    return model


def plot_prediction(model, device, plot_fields, epochs):
    test = get_dataset("test", PATH_TO_BASE_HOME, normalize=True)
    test_loader = DataLoader(test, 1, shuffle=True) 
    with torch.no_grad():
        batch = next(iter(test_loader))
        vrmse = VRMSE()
        x = batch["input_fields"]
        x = x.to(device)
        x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
        y = batch["output_fields"]
        y = y.to(device)
        # y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")
        yp = model(x)
        yp = rearrange(yp, "B C Lx Ly -> B 1 Lx Ly C")
        # print(test.metadata.n_spatial_dims)
        print(vrmse.eval(yp, y, test.metadata).mean())
        yp = rearrange(yp, "b f h w c -> b (f c) h w")
        y = rearrange(y,  "b f h w c -> b (f c) h w")
        
        print(f"mse inp tar {(x-y).square().mean()}")
        print(f"mse tar pred {(y-yp).square().mean()}")

        x = x[0].detach().numpy()
        y = y[0].detach().numpy()
        yp = yp[0].detach().numpy()
        
       
        fig, axs = plt.subplots(3, 3, figsize=(3 * 2.1, 3 * 1.2))
        for field in range(3):
            a = np.abs(x[field] - y[field])
            b = np.abs(y[field]-yp[field])
            c = np.abs(yp[field]-x[field])
            diffs = [a, b, c]
            gmax = max(a.max(), b.max(), c.max())
            for i in range(3):
                axs[field, i].imshow(diffs[i], cmap="RdBu_r", interpolation="none", vmin=0, vmax=gmax)
                axs[field, i].set_xticks([])
                axs[field, i].set_yticks([])
        axs[0, 0].set_title("|X-Y|")
        axs[0, 1].set_title("|Y-Y'|")
        axs[0, 2].set_title("|X-Y'|")
        axs[0, 0].set_ylabel("height")
        axs[1, 0].set_ylabel("velocity theta")
        axs[2, 0].set_ylabel("velocity phi")
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, f"{epochs}epochs_train.pdf"), bbox_inches='tight', dpi=200)
        plt.show()
        if plot_fields:
            fig, axs = plt.subplots(3, 3) 
            for field in range(3):
                axs[field, 0].imshow(x[field], cmap="RdBu_r", interpolation="none")
                axs[field, 1].imshow(y[field], cmap="RdBu_r", interpolation="none")
                axs[field, 2].imshow(yp[field], cmap="RdBu_r", interpolation="none")
                axs
            axs[0, 0].set_title("Input")
            axs[0, 1].set_title("Target")
            axs[0, 2].set_title("Prediction")
            axs
            plt.show()
            
            

def plot_error_after_n_steps(model, n, device):
    """"There seems to be an issue with the next step at the moment since x is not y of the 
    previous step..."""
    test = get_dataset("test", PATH_TO_BASE_HOME, normalize=True)
    test_loader = DataLoader(test, 1, shuffle=False) # We do not shuffle to be able to see the developement of error.
    with torch.no_grad():
        batch = next(iter(test_loader))
        x = batch["input_fields"]
        x = x.to(device)
        x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
        y = batch["output_fields"]
        y = y.to(device)
        y = rearrange(y, "B Ti Lx Ly F -> B (Ti F) Lx Ly") 
        xtr = x
        diffxpred = []
        diffypred = []
        mse_grad = []
        for i, batch in enumerate(test_loader):
            if i != 0:
                x = model(x)
                print(f"Diff x pred {(xtr-x).square().mean()}")
                mse_grad.append((xtr-y).square().mean().detach().item())
                diffxpred.append((xtr-x).square().mean().detach().item())
                xtr = batch["input_fields"]
                xtr = xtr.to(device)
                xtr = rearrange(xtr, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                diffypred.append((xtr-x).square().mean().detach().item())
                print(f"diff should be zero {(y-xtr).square().mean()}")
                print(f"error out after {i+1} steps {(y-x).square().mean()}")
                print(f"error after {i+1} steps: {(xtr-x).square().mean()}")
                y = batch["output_fields"]
                y = y.to(device)
                y = rearrange(y, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
            if i == n:
                break
        fig, axs = plt.subplots(1,3, figsize=(3 * 5, 2.5))
        x = x[0].detach().numpy()
        xtr = xtr[0].detach().numpy()
        for field in range(3):
            diff = np.abs(x[field] - xtr[field])
            gmax = diff.max()
            axs[field].imshow(diff, cmap="RdBu_r", interpolation="none", vmin=0, vmax=gmax)
            axs[field].set_xticks([])
            axs[field].set_yticks([])
        axs[0].set_title("height")
        axs[1].set_title("velocity theta")
        axs[2].set_title("velocity phi")
        fig.suptitle(f"Difference pred, inp after {n} steps")
        plt.tight_layout()
        plt.savefig(f"images/ErrorAfter{n}Steps.pdf", bbox_inches='tight', dpi=200)
        plt.show()
        plt.figure()
        xax = list(range(1, n+1))
        plt.plot(xax,diffxpred, label="mse(x, yp)")
        plt.plot(xax,diffypred, label="mse(y, yp)")
        plt.plot(xax,mse_grad, label="mse(gradient)")
        plt.ylabel("Error")
        plt.xlabel("Iteration")
        plt.title("Autoregressive error of predictor")
        plt.legend()
        plt.tight_layout()
        plt.savefig("images/EvolutionOfError.pdf", bbox_inches='tight', dpi=200)
        plt.show()
    
            
            


if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = train_UKAN(1, 0.01, device, bs=1, home=True, padding='asym_all')
    model = load_trained_UKAN_ptfile("recent_UNet.pt", device, KAN=False)
    # model = load_trained_UKAN_pth_file("UKAN.pth", device)
    # test_model(model, device) 
    plot_prediction(model, device, True, epochs="50")
    plot_error_after_n_steps(model, 10, device)