# %% [markdown]
# # Getting Strated with Distributed Data Parallel
# DistributedDataParallel(DDP) implements data parallelism at the module level which can run across multiple machines. Applications using DDP should spawn multiple processes and create a single DDP instance per process.
#
# The recommended way to use DDP is to spawn one process for each model replica, where a model replica can span multiple devices. **DDP processes can be placed on the same machine or across machines, but GPU devices cannot be shared across processes**

# %% [markdown]
# ## Comparison between `DataParallel` and `DistributedDataParallel`
# - `DataParallel` is single-process, multi-thread, and only works on a single machine, while `DistributedDataParallel` is multi-process and works for both single- and multi- machine training. `DataParallel` is usually slower than `DistributedDataParallel` even on a single machine due to GIL contention across threads.
# - `DistributedDataParallel` works with model parallel, `DtaaParallel` does not at this time.
# - `RPC API` for more generic distributed training support.

# %% [markdown]
# ## Basic Use Case
# To create DDP modules, first set up process groups properly. Then create a custome module wrapped with DDP and feed it with some dummy input data.
#
# *Gradient synchronization communications take place during the backward pass and overlap with the backward computation.*

# %%
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    dist.init_process_group('gloo', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)

n_gpus = torch.cuda.device_count()
assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
world_size = n_gpus
run_demo(demo_basic, world_size)

# %% [markdown]
# ## Skewed Processing Speeds
# In DDP, distributed process expected to launch and order in same number of synchronizations at roughly the same time. Otherwise, fast processes might arrive early and timeout on waiting for stragglers due to *e.g. network delays, resource contentions, unpredictable workload spikes*. **To avoid timeout in these situations, make sure that you pass a sufficiently large `timeout` value when calling `init_process_group`.**

# %% [markdown]
# ## Save and Load Checkpoints
# In DDP, one optimization is to save the model in only one process and then load it to all processes, reducing write overhead. Because all processes start from the same parameters and gradients are synchronized in backward passes. In this optimization, you need:
# - make sure all processes do net start loading before the saving in finished.
# - provide an appropriate `map_location` argument to prevent a process to step into others' devices.

# %%
def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)
    
    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5)
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DPP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()

run_demo(demo_checkpoint, world_size)

# %% [markdown]
# ## Combine DDP with Model Parallelism
# When passing a multi-GPU model to DDP, `device_ids` and `output_device` must **NOT** be set. Input and output data will be placed in proper devices by either the application or the model `forward()` method.

# %%
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)

def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = (rank * 2) % world_size
    dev1 = (rank * 2 + 1) % world_size
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

run_demo(demo_model_parallel, world_size)