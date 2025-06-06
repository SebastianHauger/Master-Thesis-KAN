"""here we add code for loading and alternatively preprocessing data """
import torch
import numpy as np
from the_well.data import WellDataset, WellDataModule
import matplotlib.pyplot as plt
from einops import rearrange



def get_dataset(partition, path_to_repo, normalize=True):
    """
    A function for getting all data.
    
    
    OUTPUT: A WellDataset which is a type of torch DataSet which is structured as 
    First we have a list of each input output pair
    Each instance contains a dictionary with field names where the only relevant 
    ones for this project are input/output fields 
    Each of these contain a tensor of shape (n_samples, 256, 512, 3)
    """
    ds = WellDataset(
        well_base_path=path_to_repo,
        well_dataset_name="planetswe",
        well_split_name=partition,
        n_steps_input=1,
        n_steps_output=1,
        use_normalization=normalize  
    )
    return ds


def get_datamodule(path_to_repo, batch_size, max_rollout_steps, normalize=True, filters=None):
    """
    A function for getting all data. 
    
    
    OUTPUT: A WellDataModule which contains batches of Data that we can use together 
    with other functions and classes from The Well in order to train models in an 
    efficient way. Note that this loads all available files (test train and valid)
    """
    if filters is not None:
        dm = WellDataModule(
            well_base_path=path_to_repo, 
            well_dataset_name="planetswe",
            batch_size=batch_size, 
            use_normalization=normalize, 
            max_rollout_steps=max_rollout_steps, 
            exclude_filters=filters
        )
    else: 
        dm = WellDataModule(
            well_base_path=path_to_repo, 
            well_dataset_name="planetswe",
            batch_size=batch_size, 
            use_normalization=normalize, 
            max_rollout_steps=max_rollout_steps
            )  
    return dm 


def plot_evolution(ds, field_names, tt=1007):
    F = ds.metadata.n_fields
    x = ds[1]["input_fields"]
    x = rearrange(x, "T Lx Ly F -> F T Lx Ly")
    fig, axs = plt.subplots(F, 4, figsize=(4 * 2.2, F * 1.2))

    for field in range(F):
        vmin = np.nanmin(x[field])
        vmax = np.nanmax(x[field])

        axs[field, 0].set_ylabel(f"{field_names[field]}")

        for i,t in enumerate([0, int(tt/3), int(2*tt/3), tt]):
            axs[field, i].imshow(
                x[field, t], cmap="RdBu_r", interpolation="none", vmin=vmin, vmax=vmax
            )
            axs[field, i].set_xticks([])
            axs[field, i].set_yticks([])

            axs[0, i].set_title(f"$t={t}$")

    plt.tight_layout()
    plt.savefig(f"images/evolution_1year.pdf",bbox_inches="tight", dpi=600)
    plt.show()
    
    