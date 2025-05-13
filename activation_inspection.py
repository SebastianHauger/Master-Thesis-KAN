import torch
import matplotlib.pyplot as plt
from ODE_KAN import KAN
import torch.nn.functional as F


def plot_activations(kan, title,  name=""):
    # axis = torch.linspace(-2, 2, 50)
    kan.eval()
    layer0 = kan.layers[0]
    in_dim = layer0.in_features
    grid0 = layer0.grid
    hidden_dim = layer0.out_features
    
    plt.rcParams['xtick.labelsize'] = 6  # Set x-axis tick size
    plt.rcParams['ytick.labelsize'] = 6  # Set y-axis tick size
    
    
    fig, ax = plt.subplots(in_dim, hidden_dim, sharex=False, sharey=False)
    for i in range(in_dim):
        X = torch.zeros((50, in_dim))
        axis = torch.linspace(grid0[i, 0], grid0[i,-1], 50)
        X[:, i] = axis 
        Xout = kan.normalizations[0](layer0(X)).detach().numpy()
        X = X.detach().numpy()
        for j in range(hidden_dim):
            ax[i,j].plot(X[:, i], Xout[:, j], color="k")
            # ax[i, j].set_aspect('equal')
            ax[i,j].grid()
    fig.suptitle("Input- to hidden layer activations. " + title)
    fig.supylabel("Input index")
    fig.supxlabel("Hidden index")
    
    fig.tight_layout()
    plt.savefig("images/Lorenz/Activations/" + name + "_in_hidden.pdf", dpi=200, bbox_inches="tight")
    plt.show()
    
    
    
    fig, ax = plt.subplots(hidden_dim, in_dim, sharex=False, sharey=False)
    layer1 = kan.layers[1]
    grid1 = layer1.grid
    for i in range(hidden_dim):
        X = torch.zeros((50, hidden_dim))
        # axis = torch.linspace(-40, 40, 160)
        axis = torch.linspace(grid1[i, 0], grid1[i,-1], 50)
        X[:, i] = axis 
        print(X.shape)
        Xout2 = kan.normalizations[1](layer1(X)).detach().numpy()
        X = X.detach().numpy()
        for j in range(in_dim):
            ax[i,j].plot(X[:, i], Xout2[:, j], color="k")
            ax[i,j].grid()
            # ax[i, j].set_aspect('equal')
    fig.suptitle("Hidden- to Output layer activations. "+ title)
    fig.supylabel("Hidden index")
    fig.supxlabel("Output index")
    fig.tight_layout()
    plt.savefig("images/Lorenz/Activations/"+ name+ "_hidden_out.pdf", dpi=200, bbox_inches='tight')
    plt.show()
    
    print(layer0.grid)
    print(layer1.grid)
    print(Xout2)
    print(Xout)
        




if __name__ == '__main__':
    model = KAN(layers_hidden=[3, 4, 3], grid_size=5, normalize=True)
    # trained = torch.load("TrainedModels/ODEKans/Lorenz/MODEL_6nodes_IR.pt")
    trained = torch.load("TrainedModels/ODEKans/Lorenz/1step_train/MODEL4NODES_CR_Final.pt")
    model.load_state_dict(trained["model_state_dict"])
    # print(model.normalizations[0].weight)
    # print(model.normalizations[0].bias)
    plot_activations(model, title ="CR 4 nodes", name="CR_4nodes")
    