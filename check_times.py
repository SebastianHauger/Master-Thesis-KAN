import torch
import torch.profiler
from UNET_classic import UNetClassic
from UKAN_smaller import UKAN
from the_well.data.datasets import WellMetadata

from main import get_data_helper


data = get_data_helper("train", True)
# Create the model
metadata = data.metadata
# model = UNetClassic(dim_in=3, dim_out=1, dset_metadata=metadata, init_features=32)
model = UKAN(padding='asym_all')
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
# Create a dummy input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(1, 3, 256, 512).to(device)  

# Use the profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,  # Include CUDA if using GPU
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),  # Save results for TensorBoard
    record_shapes=True,  # Record input shapes
    with_stack=True,  # Include stack traces
    profile_memory=True,  # Profile memory usage
) as prof:
    output = model(x)  # Forward pass

# Print the profiler results
print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=10))