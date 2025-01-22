"""here we add code for loading and alternatively preprocessing data """
import glob
import h5py
from torch.utils.data import DataLoader
import torch
import numpy as np
from the_well.data import WellDataset
import matplotlib.pyplot as plt
from einops import rearrange


def get_all_data(index = 3) -> torch.Tensor:
    """a function that later will be for getting all data. At the moment we only use it 
    to get a single frame."""
    path_to_repo = "datasets"
    ds = WellDataset(
        well_base_path=path_to_repo,
        well_dataset_name="planetswe",
        well_split_name="test",
        n_steps_input=1000,
        n_steps_output=8,
        use_normalization=False
    )
    return ds


def plot_an_image_frame(ds):
    F = ds.metadata.n_fields
    x = ds[42]["input_fields"]
    x = rearrange(x, "T Lx Ly F -> F T Lx Ly")

    fig, axs = plt.subplots(F, 4, figsize=(4 * 2.4, F * 1.2))

    for field in range(F):
        vmin = np.nanmin(x[field])
        vmax = np.nanmax(x[field])

        axs[field, 0].set_ylabel(f"{field_names[field]}")

        for t in range(4):
            axs[field, t].imshow(
                x[field, t], cmap="RdBu_r", interpolation="none", vmin=vmin, vmax=vmax
            )
            axs[field, t].set_xticks([])
            axs[field, t].set_yticks([])

            axs[0, t].set_title(f"$x_{t}$")

    plt.tight_layout()
    plt.show()
    
    


def load_datafile(a):
    train = DataLoader(a, 2008, False)
    return train


if __name__=='__main__':
    ds = get_all_data()
    # dsl = load_datafile(ds)
    item = ds[0]
    keyss = list(item.keys()) 
    for key in keyss:
        print(f"key: {key} shape: {item[key].shape}")
    field_names = [name for group in ds.metadata.field_names.values() for name in group]
    print (field_names)
    print(len(ds))
    plot_an_image_frame(ds)