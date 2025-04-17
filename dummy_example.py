from ODE_KAN import KAN
from cycler import cycler
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm 
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib as mpl

mpl.rcParams['lines.color'] = "black"
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.prop_cycle'] = cycler(color=["black", "red", "orange", "green"])


def xy_data(n_x=20, n_y=20):
    x = torch.linspace(-1, 1, n_x)
    y = torch.linspace(-1, 1, n_y)
    X,Y = torch.meshgrid(x,y, indexing="ij")
    Z = X * Y 
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    print(X.shape, Y.shape, Z.shape)
    return X, Y, Z

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 1))

    def forward(self, x):
        return self.net(x)

class XYDataset(Dataset):
    def __init__(self, inputs, targets):
        super().__init__()
        self.inputs = torch.stack(inputs, dim=-1)
        self.targets = targets
    
    
    def __len__(self):
        return self.inputs.shape[0]
    
    
    def __getitem__(self, index):
        
        return self.inputs[index], self.targets[index]
    
        
        


def train_model(X, Y, Z, epochs):
    model = KAN([2, 2, 1], grid_size=5, grid_range=[-1.5, 1.5])
    # model = MLP()
    print(model.layers[0].spline_weight.shape)
    print(model.layers[1].spline_weight.shape)
    ds = XYDataset((X,Y), Z)
    dL = DataLoader(ds, batch_size=25, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)
    loss = nn.MSELoss()
    loss_list = []
    model.train()
    for epoch in (bar:= tqdm(range(epochs))):        
        ep_loss = 0
        
        for input, target in dL:
            
            # print(input)
            # print(target)
            pred=model(input, update_grid=False)
            # if epoch % 100 == 0:
            #     pred = model(input, update_grid=True)
            
            loss = torch.mean(torch.square(target-pred.transpose(0,1))) 
            
            # print((target-pred.transpose(0,1)).shape)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
            # print(loss.item())
            ep_loss += loss.item()
            
        
        scheduler.step()
        
        bar.set_postfix(loss=(ep_loss))
        loss_list.append(ep_loss)
    # print(model.net[0].weight)
    plt.figure()
    plt.semilogy(loss_list)
    plt.show()
    
    print(plt.rcParams['lines.color'])
    
    inputs = ds.inputs
    inp = inputs.detach().numpy()
    layer = model.layers[0]
    splines = layer.b_splines(inputs)
    print(inputs.shape)
    print(splines.shape)
    print(layer.scaled_spline_weight.shape)
    # Xout = nn.functional.linear(splines.view(inputs.size(0), -1), layer.scaled_spline_weight.view(2, -1))
    # Xout = Xout.detach().numpy()
    Xout = layer(inputs).detach().numpy()
    
    X = inp[:, 0].reshape((10, 10))
    Y = inp[:, 1].reshape((10, 10))
    # print(f"X = {X}")
    # print(f"Y = {Y}")
    fig, ax = plt.subplots(2, 1, subplot_kw={"projection": "3d"})
    for i in range(2):
        
        Z = Xout[:, i].reshape((10, 10))
        # print(f"Z = {Z}")
        ax[0].plot_surface(X, Y, Z, label=f"node {i}")
        
        
    print(model.layers[0].spline_weight.shape)
    print(model.layers[1].spline_weight.shape)
    
    # now show the expected shapes: 
    print(layer.base_weight)
    first = (X + Y)
    second = (X**2 + Y**2)
    ax[1].plot_surface(X, Y, first, label="X + Y")
    ax[1].plot_surface(X,Y, second, label = r"$X^2 + Y^2$")
    ax[0].legend()
    ax[1].legend()
    
    
    plt.show()
    
    empty = torch.zeros(50)
    xy_short = torch.linspace(-1, 1, 50)
    
    xdat = torch.stack((xy_short, empty), dim=-1)
    ydat = torch.stack((empty, xy_short), dim=-1)
    
    x_act = layer(xdat).detach().numpy()
    y_act = layer(ydat).detach().numpy()
    print(x_act.shape)
    
    fig, ax = plt.subplots(2, 2, sharex=True)
    xax = xdat[:, 0].detach().numpy()
    print(xdat.shape)
    print(xax)
    ax[0, 0].plot(xax, x_act[:,0])
    ax[1, 0].plot(xax, x_act[:,1])
    ax[0, 1].plot(xax, y_act[:,0])
    ax[1, 1].plot(xax, y_act[:,1])
    fig.supxlabel("Input dim")
    fig.supylabel("Hidden dim")
    plt.tight_layout()
    plt.savefig("images/Misc/XY_act_inHidden.pdf", dpi=200, bbox_inches='tight')
    plt.show()
    
    
    
    fig, ax = plt.subplots(2, 1, sharex=True)
    layer2 = model.layers[1]
    x_act2 = layer2(xdat).detach().numpy()
    y_act2 = layer2(ydat).detach().numpy()
    ax[0].plot(xax, x_act2)
    ax[1].plot(xax, y_act2)
    fig.supylabel("Hidden dim")
    fig.supxlabel("Output dim")
    plt.savefig("images/Misc/XY_act_hiddenOut.pdf", dpi=200, bbox_inches='tight')
    plt.show()
    
 

    target = ds.targets
    predictions = model(inputs)
    
    
    
    
    pred = predictions.detach().numpy()
    X = inp[:, 0].reshape((10, 10))
    Y = inp[:, 1].reshape((10, 10))
    Z = pred.reshape((10, 10))
    Z_true = target.detach().numpy().reshape((10,10))
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.plot_surface(X, Y, Z)
    ax.plot_surface(X,Y, Z_true)
    plt.show()
    

if __name__=="__main__":
    X, Y, Z = xy_data(10, 10)
    train_model(X, Y, Z, 1000)
    
    
        
            