"""root of the simulation. Run files from here."""
from torch.utils.data import DataLoader
from UKAN import UKAN
from tqdm import tqdm 
from data import get_data
from einops import rearrange
import torch


def train_UKAN(epochs, lr):
    PATH_TO_BASE_HOME = "datasets"
    PATH_TO_BASE_ALVIS = "/mimer/NOBACKUP/groups/shallow_ukan/datasets"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    train = get_data("train", PATH_TO_BASE_ALVIS)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    model=UKAN()
    model=model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for _ in range(epochs):
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
            optimizer.zero_grad()
            bar.set_postfix(loss=mse.detach().item())
        
            
    

if __name__=='__main__':
    train_UKAN(2, 0.01)