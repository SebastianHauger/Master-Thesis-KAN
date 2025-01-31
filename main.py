"""root of the simulation. Run files from here."""
from torch.utils.data import DataLoader
from UKAN import UKAN
from tqdm import tqdm 
from data import get_data
from einops import rearrange
import torch
import matplotlib.pyplot as plt
import numpy as np 


PATH_TO_BASE_HOME = "datasets"
PATH_TO_BASE_ALVIS = "/mimer/NOBACKUP/groups/shallow_ukan/datasets"


def train_UKAN(epochs, lr, device, bs=64):
    print(device)
    train = get_data("train", PATH_TO_BASE_ALVIS)
    val = get_data("valid", PATH_TO_BASE_ALVIS)
    train_loader = DataLoader(train, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val, batch_size=bs, shuffle=False)
    model=UKAN()
    model=model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for i in tqdm(range(epochs)):
        for batch in (bar := tqdm(train_loader)):
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
        with torch.no_grad():
            avg_mse=0
            for n,batch in enumerate(val_loader):
                x = batch["input_fields"]
                x = x.to(device)
                x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                y = batch["output_fields"]
                y = y.to(device)
                y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")
                yp = model(x)
                mse = (yp-y).square().mean()
                avg_mse = (avg_mse*n + mse)/(n+1) # mooving average MSE
            print(f"validation loss epoch {i}: {avg_mse}")
    torch.save(model.state_dict(), 'UKAN.pth')
    return model 
                
                 
def test_model(model, device, bs=64):
    test = get_data("test", PATH_TO_BASE_HOME)
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
            
            
def load_trained_UKAN(device):
    """assuming model has been trained with cuda."""
    model = UKAN()
    if device == 'cpu':
        model.load_state_dict(torch.load('Trained models/UKAN.pth', map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load('Trained models/UKAN.pth'))
    model.to(device)
    return model


def plot_prediction(model, device):
    test = get_data("test", PATH_TO_BASE_HOME)
    test_loader = DataLoader(test, 1, shuffle=False) 
    with torch.no_grad():
        batch = next(iter(test_loader))
        x = batch["input_fields"]
        x = x.to(device)
        x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
        y = batch["output_fields"]
        y = y.to(device)
        y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")
        yp = model(x)
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


if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_UKAN(2, 0.01, device)
    # model = load_trained_UKAN(device)
    test_model(model, device) 
    # plot_prediction(model, device)