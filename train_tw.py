from the_well.benchmark.trainer.training import Trainer
from the_well.benchmark.metrics.spatial import VRMSE
from torch.utils.data import DataLoader
from STD_UKAN import UKAN
from tqdm import tqdm 
from data import get_data
from einops import rearrange
import torch
import matplotlib.pyplot as plt
import numpy as np 


INFO = {
    "alvis": {
        "cpf": "TrainedModels", 
        "af": "Artifacts",
        "viz": "Vizualisation",
        "cp_freq": 5, 
        "val_freq": 2, 
        "rollout_val_freq": 5,
        "max_rollout_steps": 5, 
        "short_validation_length": 20, 
        "num_time_intervals": 1
        }, 
    "home" : {
        "cpf": "TrainedModels", 
        "af": "Artifacts",
        "viz": "Vizualisation",
        "cp_freq": 5, 
        "val_freq": 2, 
        "rollout_val_freq": 5,
        "max_rollout_steps": 5, 
        "short_validation_length": 20, 
        "num_time_intervals": 1
        }
    }

def train_and_eval(checkpoint_folder, artifact_folder, viz_folder, formatter, 
                   checkpoint_frequency, val_frequency, rollout_val_frequency, 
                   max_rollout_steps, short_validation_length, num_time_intervals, 
                   device, checkpoint_path="", padding='uniform'):
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
    model = UKAN(padding=padding)
    loss = VRMSE()   # Use same loss as them
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    tr = Trainer(checkpoint_folder=checkpoint_folder,
                 artifact_folder=artifact_folder, 
                 viz_folder=viz_folder, 
                 formatter=formatter, 
                 model=model, 
                 datamodule=None, 
                 loss_fn=loss.forward, 
                 epochs = 1,
                 checkpoint_frequency=checkpoint_frequency,
                 val_frequency=val_frequency,
                 rollout_val_frequency= rollout_val_frequency, 
                 max_rollout_steps=max_rollout_steps,
                 short_validation_length=short_validation_length,
                 make_rollout_videos=False,
                 num_time_intervals=num_time_intervals, 
                 lr_scheduler=None,
                 device = device, 
                 is_distributed=False, 
                 checkpoint_path=checkpoint_path
                 )
    tr.train()
    return model


if __name__=='__main__':
    checkpoint_folder = "/Trained models"
    