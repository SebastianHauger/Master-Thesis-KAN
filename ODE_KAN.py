import torch
import torch.nn as nn
from KANlayers import KANLinear




class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        normalize=False
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.normalizations = torch.nn.ModuleList()
        if normalize:
            for i in range(len(layers_hidden)-2):
                self.normalizations.append(torch.nn.BatchNorm1d(num_features=layers_hidden[1+i]))
            self.normalizations.append(torch.nn.Identity())
        else: 
            for i in range(len(layers_hidden)-1):
                self.normalizations.append(torch.nn.Identity())
        
            

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer, norm in zip(self.layers, self.normalizations):
            if update_grid:
                layer.update_grid(x)
            x = norm(layer(x)) 
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )