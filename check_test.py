"""root of the simulation. Run files from here."""
from torch.utils.data import DataLoader
from STD_UKAN import UKAN
from tqdm import tqdm 
from data import get_dataset, get_datamodule
from einops import rearrange
import torch
import matplotlib.pyplot as plt
import numpy as np 
import random 
from the_well.benchmark.metrics import VRMSE
from the_well.data.data_formatter import DefaultChannelsFirstFormatter


PATH_TO_BASE_HOME = "datasets"
PATH_TO_BASE_ALVIS = "/mimer/NOBACKUP/groups/shallow_ukan/datasets"

def get_data_helper(split, home):
    if home:
        split="test"
        return get_dataset(split, PATH_TO_BASE_HOME)
    else:
        return get_dataset(split, PATH_TO_BASE_ALVIS)
      
            
def load_trained_UKAN(name,device):
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


def plot_prediction(model, device, plot_fields):
    dm = get_datamodule(PATH_TO_BASE_HOME, 1, 5, filters=["train", "valid"])
    test_loader = dm.test_dataloader()
    formatter = DefaultChannelsFirstFormatter()
    model.test()
    with torch.no_grad():
        batch = next(iter(test_loader))
        input, y = formatter.process_input(batch)
        
        x = batch["input_fields"]
        x = x.to(device)
        x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
        y = batch["output_fields"]
        y = y.to(device)
        y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")
        yp = model(x)
        yp = formatter.process_output(yp)
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
    # model = train_UKAN(1, 0.01, device, bs=1, home=True, padding='asym_all')
    model = load_trained_UKAN("checkpoint_50.pt", device)
    # test_model(model, device) 
    plot_prediction(model, device, True)