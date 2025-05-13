"""root of the simulation. Run files from here."""
from torch.utils.data import DataLoader
from new_UKAN_proposal import UKAN
# from the_well.benchmark.models.unet_classic import UNetClassic
from the_well.data import WellDataset, WellDataModule
from UNET_classic import UNetClassic
from tqdm import tqdm 
from data import get_dataset, get_datamodule
from einops import rearrange
import torch
import matplotlib.pyplot as plt
import numpy as np 
import random 
from the_well.benchmark.metrics import VRMSE
import os
from UKAN_smaller import UKAN as SmallUKAN
from itertools import islice


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
    # model = UNetClassic(dim_in=3, dim_out=3, dset_metadata=train.metadata, init_features=32)
    model = SmallUKAN(padding=padding)
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
            
            
def load_trained_UKAN_ptfile(name,device, KAN, large=True):
    """assuming model has been trained with cuda."""
    if KAN:
        if large:
            model = UKAN(padding="asym_all")
        else:
            model = SmallUKAN(padding='uniform')
    else:
        dataset = get_dataset("train", PATH_TO_BASE_HOME, True)
        model = UNetClassic(3, 3, dataset.metadata)
    if device == 'cpu':
        checkpoint = torch.load(os.path.join(MODEL_DIR,name), map_location=torch.device('cpu'), weights_only=False)
    else:
        checkpoint = torch.load(os.path.join(MODEL_DIR,name), weights_only=False)
    print(checkpoint.keys())
    print(checkpoint["validation_loss"])
    # print(checkpoint["train_loss"])
    print(checkpoint["epoch"])
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
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


def plot_prediction(model, device, title, name):
    test = get_dataset("test", PATH_TO_BASE_HOME, normalize=True)
    test_loader = DataLoader(test, 1, shuffle=False)
    model.eval() 
    with torch.no_grad():
        # batch = next(islice(test_loader, 400, 401))
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
        print(vrmse.eval(yp, y, test.metadata))
        yp = rearrange(yp, "b f h w c -> b (f c) h w")
        y = rearrange(y,  "b f h w c -> b (f c) h w")
        
        print(f"mse inp tar {(x-y).square().mean()}")
        print(f"mse tar pred {(y-yp).square().mean()}")

        x = x[0].detach().numpy()
        y = y[0].detach().numpy()
        yp = yp[0].detach().numpy()
        fig, axs = plt.subplots(3, 3, figsize=(3 * 2.1, 3 * 1.2))
        for field in range(3):
            a = np.abs(x[field]-y[field])
            b = np.abs(y[field]-yp[field])
            c = np.abs(yp[field]-x[field])
            diffs = [a, b, c]
            gmax = max(a.max(), b.max(), c.max())
            for i in range(3):
                axs[field, i].imshow(diffs[i], cmap="RdBu_r", interpolation="none", vmin=0, vmax=gmax)
                axs[field, i].set_xticks([])
                axs[field, i].set_yticks([])
        axs[0, 0].set_title("|X-Y|", fontsize=14)
        axs[0, 1].set_title("|Y-Y'|", fontsize=14)
        axs[0, 2].set_title("|X-Y'|", fontsize=14)
        axs[0, 0].set_ylabel("height", fontsize=9)
        axs[1, 0].set_ylabel("velocity theta", fontsize=9)
        axs[2, 0].set_ylabel("velocity phi", fontsize=9)
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, name+"_gradients.pdf"), bbox_inches='tight', dpi=200)
        plt.show()
            
            

def plot_error_after_n_steps(model1, model2, model3, n, device, times, name="Large_UKAN_"):
    """"There seems to be an issue with the next step at the moment since x is not y of the 
    previous step..."""
    test = get_dataset("test", PATH_TO_BASE_HOME, normalize=True)
    test_loader = DataLoader(test, 1, shuffle=False) # We do not shuffle to be able to see the developement of error.
    model1.eval()
    model2.eval()
    model3.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        x1 = batch["input_fields"]
        x1 = x1.to(device)
        x1 = rearrange(x1, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
        y = batch["output_fields"]
        y = y.to(device)
        y = rearrange(y, "B Ti Lx Ly F -> B (Ti F) Lx Ly") 
        xtr = x1
        diffypred1 = []
        diffypred2 = []
        diffypred3 = []
        mse_grad = []
        fields = []
        plotx1 = []
        plotx2 = []
        plotx3 = []
        x2 = x1
        x3 = x1
     
        
        for i, batch in enumerate(islice(test_loader, 0, n+1)):
            if i != 0:
                x1 = model1(x1)
                x2 = model2(x2)
                x3 = model3(x3)
                print(f"Diff x pred {(xtr-x1).square().mean()}")
                xtr = batch["input_fields"]
                xtr = xtr.to(device)
                xtr = rearrange(xtr, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                diffypred1.append((xtr-x1).square().mean().detach().item())
                diffypred2.append((xtr-x2).square().mean().detach().item())
                diffypred3.append((xtr-x3).square().mean().detach().item())
                print(f"diff should be zero {(y-xtr).square().mean()}")
                print(f"error out after {i+1} steps {(y-x1).square().mean()}")
                print(f"error after {i+1} steps: {(xtr-x1).square().mean()}")
                y = batch["output_fields"]
                y = y.to(device)
                y = rearrange(y, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                if i in times:
                    fields.append(xtr[0].detach().numpy())
                    plotx1.append(x1[0].detach().numpy())
                    plotx2.append(x2[0].detach().numpy())
                    plotx3.append(x3[0].detach().numpy())
                    
                    
        names = ["Height", "Velocity Theta", "Velocity Phi"]
        name_F = ["height", "vel_t", "vel_p"] 
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for field in range(3):
            fig, axs = plt.subplots(4, len(fields), figsize=(len(fields)*6.1, 13.2))
            for i in range(len(fields)):
                axs[0, i].imshow(plotx1[i][field], cmap='RdBu_r', interpolation=None)
                axs[1, i].imshow(plotx2[i][field], cmap='RdBu_r', interpolation=None)
                axs[2, i].imshow(plotx3[i][field], cmap='RdBu_r', interpolation=None)
                axs[3, i].imshow(fields[i][field], cmap='RdBu_r', interpolation=None)
                for j in range(4):
                    axs[j, i].set_xticks([])
                    axs[j, i].set_yticks([])
                    
                axs[0, i].set_title(f"t = {times[i]}", fontsize=36)
                
                axs[0, i].set_yticks([])
                axs[1, i].set_yticks([])  
                
            axs[0, 0].set_ylabel(f"S UKAN", fontsize=36)
            axs[1, 0].set_ylabel(f"U-net", fontsize=36)
            axs[2, 0].set_ylabel(f"L UKAN", fontsize=36)
            axs[3, 0].set_ylabel(f"GT", fontsize=36)
            fig.suptitle(names[field], fontsize=36)
            plt.tight_layout()
            fig.savefig(os.path.join(IMAGE_DIR, name+name_F[field]+"_rollout.pdf"), dpi=200, bbox_inches='tight')
            plt.show()
        plt.figure()
        xax = list(range(1, n+1))
        plt.plot(diffypred1, linewidth=0.8, color="k", label="Small UKAN")
        plt.plot(diffypred2, linewidth=0.8, color='r', label="U-net")
        plt.plot(diffypred3, linewidth=0.8, color='b', label="Large UKAN")
        plt.title("Rollout error growth", fontsize=16)
        plt.ylabel("MSE", fontsize=14)
        plt.xlabel("Rollout step", fontsize=14)
        plt.ylim((-0.5, 2.0))
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR,"rollout_error_growth.pdf"), dpi=200, bbox_inches='tight')
        plt.show()
        
        
        
        
            
            
def get_num_trainable_parameters(model):
    # get the number of trainable parameters. 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  


if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = UNetClassic(3, 3, )
    # model = train_UKAN(1, 0.01, device, bs=1, home=True, padding='asym_all')
    unet = load_trained_UKAN_ptfile("recent_UNet.pt", device, KAN=False)
    large_ukan = load_trained_UKAN_ptfile("recent_UKAN_large.pt", device, KAN=True)
    small_ukan = load_trained_UKAN_ptfile("recent_UKAN_small.pt", device, KAN=True, large=False)
    # model = SmallUKAN(padding='asym_all')
    # print(get_num_trainable_parameters(model))
    # model = load_trained_UKAN_pth_file("UKAN.pth", device)
    # test_model(model, device) 
    # plot_prediction(unet, device, "U-net gradient prediction", name="U-net")
    plot_error_after_n_steps(small_ukan, unet, large_ukan, 300, device,  [25, 100 , 200, 300], "comparrison_")