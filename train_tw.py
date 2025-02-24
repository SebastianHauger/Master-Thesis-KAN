from the_well.benchmark.trainer.training import Trainer
from the_well.benchmark.metrics.spatial import VRMSE
from torch.utils.data import DataLoader
from new_UKAN_proposal import UKAN
from tqdm import tqdm 
from data import get_datamodule
from einops import rearrange
import torch
import matplotlib.pyplot as plt
import numpy as np 
import wandb


# Note that we stream from the hugging face platfrom while running on
# home computer since we don't have all splits downloaded.
INFO = {
    "alvis": {
        "cpf": "TrainedModels", 
        "af": "Artifacts",
        "viz": "Vizualisation",
        "formatter": "channels_first_default",
        "cp_freq": 5, 
        "val_freq": 2, 
        "rollout_val_freq": 5,
        "max_rollout_steps": 1, 
        "short_validation_length": 20, 
        "num_time_intervals": 1,
        "ptb": "/mimer/NOBACKUP/groups/shallow_ukan/datasets",
        "cpp": "TrainedModels/recent.pt"  # load the most recent model.
        }, 
    "home" : {
        "cpf": "TrainedModels", 
        "af": "Artifacts",
        "viz": "Vizualisation",
        "formatter": "channels_first_default",
        "cp_freq": 5, 
        "val_freq": 2, 
        "rollout_val_freq": 5,
        "max_rollout_steps": 5, 
        "short_validation_length": 20, 
        "num_time_intervals": 1,
        "ptb": "datasets"
        }
    }


def train_and_eval(checkpoint_folder, artifact_folder, viz_folder, formatter, 
                   checkpoint_frequency, val_frequency, rollout_val_frequency, 
                   max_rollout_steps, short_validation_length, num_time_intervals, 
                   device, path_to_base, batch_size, checkpoint_path="", 
                   padding='uniform', epochs=1, normalize=True, max_lr=0.001, min_lr=0.0001):
    """
    This part of docstring simply copied from The Well documentation for Trainer class 
    Args:
        checkpoint_folder:
            Path to folder used for storing checkpoints.
        artifact_folder:
            Path to folder used for storing artifacts.
        viz_folder:
            Path to folder used for storing visualizations.
        epochs:
            Number of epochs to train the model.
            One epoch correspond to a full loop over the datamodule's training dataloader
        checkpoint_frequency:
            The frequency in terms of number of epochs to save the model checkpoint
        val_frequency:
            The frequency in terms of number of epochs to perform the validation
        rollout_val_frequency:
            The frequency in terms of number of epochs to perform the rollout validation
        max_rollout_steps:
            The maximum number of timesteps to rollout the model
        short_validation_length:
            The number of batches to use for quick intermediate validation during training
        make_rollout_videos:
            A boolean flag to trigger the creation of videos during long rollout validation
        num_time_intervals:
            The number of time intervals to split the loss over
        device: A Pytorch device (e.g. "cuda" or "cpu")
        is_distributed:
            A boolean flag to trigger DDP training
        enable_amp:
            A boolean flag to enable automatic mixed precision training
        amp_type:
            The type of automatic mixed precision to use. Can be "float16" or "bfloat16"
        checkpoint_path:
            The path to the model checkpoint to load. If empty, the model is trained from scratch.
    """
    wandb.init()
    model = UKAN(padding=padding).to(device)
    loss = VRMSE()   # Use same loss as them
    datamodule = get_datamodule(path_to_repo=path_to_base,
                                batch_size=batch_size, 
                                max_rollout_steps=max_rollout_steps, 
                                normalize=normalize)
    
    
    optim = torch.optim.Adam(model.parameters(), lr=max_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=int(epochs/2 + 0.1), eta_min=min_lr, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 5, 0.2)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.9)
    tr = Trainer(checkpoint_folder=checkpoint_folder,
                 artifact_folder=artifact_folder, 
                 viz_folder=viz_folder, 
                 formatter=formatter, 
                 model=model, 
                 datamodule=datamodule, 
                 loss_fn=loss.eval, 
                 epochs = epochs,
                 optimizer=optim,
                 checkpoint_frequency=checkpoint_frequency,
                 val_frequency=val_frequency,
                 rollout_val_frequency= rollout_val_frequency, 
                 max_rollout_steps=max_rollout_steps,
                 short_validation_length=short_validation_length,
                 make_rollout_videos=False,
                 num_time_intervals=num_time_intervals, 
                 lr_scheduler=scheduler,
                 device = device, 
                 is_distributed=False, 
                 checkpoint_path=checkpoint_path
                 
                 )
    tr.train()
    wandb.finish()


if __name__=='__main__':
    # print(torch.__version__)
    b = "alvis"
    info = INFO[b]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    padding = 'asym_all'
    train_and_eval(
        checkpoint_folder=info["cpf"], 
        artifact_folder=info["af"],
        viz_folder=info["viz"],
        formatter=info["formatter"], 
        checkpoint_frequency=info["cp_freq"], 
        val_frequency=info["val_freq"], 
        rollout_val_frequency=info["rollout_val_freq"],
        max_rollout_steps=info["max_rollout_steps"],
        short_validation_length=info["short_validation_length"],
        num_time_intervals=info["num_time_intervals"], 
        device=device, 
        padding=padding, 
        epochs=50,
        path_to_base=info["ptb"],
        batch_size=54, 
        normalize=True,
        checkpoint_path=info["cpp"] #remove this if one wishes to train from beginning
        )
    
    