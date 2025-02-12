"""root of the simulation. Run files from here."""
from torch.utils.data import DataLoader
from STD_UKAN import UKAN
from tqdm import tqdm 
from data import get_dataset
from einops import rearrange
import torch
import matplotlib.pyplot as plt
import numpy as np 
import random 
from the_well.benchmark.metrics import VRMSE


PATH_TO_BASE_HOME = "datasets"
PATH_TO_BASE_ALVIS = "/mimer/NOBACKUP/groups/shallow_ukan/datasets"

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
    model=UKAN(padding=padding)
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
            if i == 100:
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
            
            
def load_trained_UKAN_ptfile(name,device):
    """assuming model has been trained with cuda."""
    model = UKAN(padding="asym_all")
    if device == 'cpu':
        checkpoint = torch.load("Trained models/"+name, map_location=torch.device('cpu'), weights_only=False)
    else:
        checkpoint = torch.load('Trained models/UKAN.pth', weights_only=False)
    print(checkpoint.keys())
    print(checkpoint["validation_loss"])
    print(checkpoint["epoch"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model

def load_trained_UKAN_pth_file(name, device):
    model = UKAN(padding="uniform")
    if device == "cpu":
        checkpoint = torch.load("Trained models/"+name, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load('Trained models/UKAN.pth')
    model.load_state_dict(checkpoint)
    model.to(device)
    return model


def plot_prediction(model, device, plot_fields):
    test = get_dataset("test", PATH_TO_BASE_HOME, normalize=True)
    test_loader = DataLoader(test, 1, shuffle=False) 
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

        x = x[0].detach().numpy()
        y = y[0].detach().numpy()
        yp = yp[0].detach().numpy()
        fig, axs = plt.subplots(3, 3) 
        for field in range(3):
            axs[field, 0].imshow(
                np.abs(x[field] - y[field]), cmap="RdBu_r", interpolation="none")
            axs[field, 1].imshow(np.abs(y[field]-yp[field]), cmap="RdBu_r", interpolation="none")
            axs[field, 2].imshow(np.abs(yp[field]-x[field]), cmap="RdBu_r", interpolation="none")
        axs[0, 0].set_title("Diff inp tar")
        axs[0, 1].set_title("Diff tar pred")
        axs[0, 2].set_title("diff pred inp")
        plt.savefig("images/2epochs_train.pdf", bbox_inches='tight', dpi=200)
        plt.show()
        if plot_fields:
            fig, axs = plt.subplots(3, 3) 
            for field in range(3):
                axs[field, 0].imshow(x[field], cmap="RdBu_r", interpolation="none")
                axs[field, 1].imshow(y[field], cmap="RdBu_r", interpolation="none")
                axs[field, 2].imshow(yp[field], cmap="RdBu_r", interpolation="none")
            axs[0, 0].set_title("Input")
            axs[0, 1].set_title("Target")
            axs[0, 2].set_title("Prediction")
            plt.show()
            
            


if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_UKAN(1, 0.01, device, bs=1, home=True, padding='asym_all')
    # model = load_trained_UKAN_pth_file("ukan.pth", device)
    # test_model(model, device) 
    # plot_prediction(model, device, True)