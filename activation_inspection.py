import torch
import matplotlib.pyplot as plt
from ODE_KAN import KAN
import torch.nn.functional as F


def plot_activations(kan):
    axis = torch.linspace(-2, 2, 50)
    layer0 = kan.layers[0]
    in_dim = layer0.in_features
    hidden_dim = layer0.out_features
    
    
    fig, ax = plt.subplots(in_dim, hidden_dim, sharex=True, sharey=True)
    for i in range(in_dim):
        X = torch.zeros((50, in_dim))
        X[:, i] = axis 
        Xout = layer0(X).detach().numpy()
        X = X.detach().numpy()
        for j in range(hidden_dim):
            ax[i,j].plot(X[:, i], Xout[:, j])
            # ax[i, j].set_aspect('equal')
    plt.show()
    
    fig, ax = plt.subplots(in_dim, hidden_dim, sharex=True, sharey=True)
    layer1 = kan.layers[1]
    for i in range(hidden_dim):
        X = torch.zeros((50, hidden_dim))
        X[:, i] = axis 
        Xout = layer1(X).detach().numpy()
        X = X.detach().numpy()
        for j in range(in_dim):
            ax[j,i].plot(X[:, i], Xout[:, j])
            # ax[i, j].set_aspect('equal')
    plt.show()
        
    
        
        
    
    layer1  = kan.layers[1]
        




if __name__ == '__main__':
    model = KAN(layers_hidden=[3, 4, 3], grid_size=5)
    trained = torch.load("TrainedModels/ODEKans/Lorenz/1step_train/checkpoint_2000.pt")
    model.load_state_dict(trained["model_state_dict"])
    plot_activations(model)
    