import torch
import matplotlib.pyplot as plt
from ODE_KAN import KAN
import torch.nn.functional as F


def plot_activations(kan):
    x = torch.arange(-10, 10.1, 0.1)
    X = x.unsqueeze(1).repeat(1, 3)
    with torch.no_grad():
        for layer in kan.layers:
            n = layer.in_features
            print(n)
            X = x.unsqueeze(1).repeat(1, n)
            # layer.base_weight = 0.0
            splines = layer.b_splines(X)
            # print(splines.shape)
            splines = splines.view(X.size(0), -1)
            # print(layer.scaled_spline_weight.shape)
            # print(layer.scaled_spline_weight)
            Xout = F.linear(splines.view(X.size(0), -1), layer.scaled_spline_weight.view(10, -1))
            # Xout.transpose(0, 1)
            
            for i in range(splines.size(1)):
                plt.figure()
                plt.plot(X[:,0].numpy(), splines[:,i].detach().numpy())
                plt.title(f"Node {i}")
                plt.show()
        




if __name__ == '__main__':
    model = KAN(layers_hidden=[3, 10, 3], grid_size=5)
    trained = torch.load("TrainedModels/ODEKans/Lorenz/checkpoint_7700.pt")
    model.load_state_dict(trained["model_state_dict"])
    plot_activations(model)
    