"""root of the simulation. Run files from here."""
from torch.utils.data import DataLoader
from UKAN import UKAN
from tqdm import tqdm 
from data import get_all_data
from einops import rearrange
import torch


def train_UKAN(epochs, lr):
    train = get_all_data()
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    model=UKAN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for _ in range(epochs):
        for batch in (bar := tqdm(train_loader)):
            x= batch["input_fields"]
            x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
            y = batch["output_fields"]
            y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")
            yp = model(x)
            mse = (yp-y).square().mean()
            mse.backward()
            optimizer.zero_grad()
            bar.set_postfix(loss=mse.detach.item())
            
    

if __name__=='__main__':
    train_UKAN(2, 0.01)