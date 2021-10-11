# %% [markdown]
# # Writing Distributed Applications with PyTorch
# `torch.distributed` can easily parallelize the computations across processes and clusters of machines. Through leverages message passing semantics allowing each process to communicate data to any of the other processes, **which can use different communication backends and are not restricted to being executed on the same machine**.

# %% [markdown]
# ## Setup
# In order to get started we need the ability to run multiple processes simultaneously.
#
# `init_process` function ensures that every process will be able to coordinate through a master, using the same ip address and port.

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from math import ceil
from random import Random
from torchvision import datasets, transforms

def run(rank, size):
    """ Distributed function to be implemented later. """
    pass

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

# %% [markdown]
# ## Point-to-Point Communication
# A transfer of data from one process to another is called a point-to-point communication. These are achieved through the `send` and `recv` functions or their *immediate* counter-parts, `isend` and `irecv`.
#
# Notice `send/recv` are **blocking**, and *immediate* `isend/irecv` are **non-blocking** which have to careful about not modify the send tensor nor access the received tensor before `req.wait()` has completed. 

# %%
"""Blocking point-to-point communication."""
def run_block_p2p(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

"""Non-blocking point-to-point communication."""
def run_nonBlock_p2p(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])

# %% [markdown]
# ## Collective Communication
# Collectives allow for communication patterns across all processes in a **group**. A group is a subset of all our processes. To create a group, we can pass a list of ranks to `dist.new_group(group)`. By default, collectives are executed on the all processes, also known as the `world`.For example, in order to obtain the sum of all tensors at all processes, using the `dist.all_reduce(tensor, op, group)` collective.
#
# PyTorch comes with 4 such operators, all working at the element-wise level:
# * `dist.ReduceOp.SUM`
# * `dist.ReduceOp.PRODUCT`
# * `dist.ReduceOp.MAX`
# * `dist.ReduceOp.MIN`
#
# There are a total of 6 collectives currently implemented in PyTorch.
# * `dist.broadcast(tensor, src, group)`: Copies `tensor` from `src` to all other processes.
# * `dist.reduce(tensor, dst, op, group)`: Applies `op` to all `tensor` and stores the result in `dst`.
# * `dist.all_reduce(tensor, op, group)`: Same as reduce, but the result is stored in all processes.
# * `dist.scatter(tensor, scatter_list, src, group)`: Copies the ith tensor `scatter_list[i]` to the ith process.
# * `dist.gather(tensor, gather_list, dst, group)`: Copies `tensor` from all processes in `dst`.
# * `dist.all_gather(tensor_list, tensor, group)`: Copies `tensor` from all processes to `tensor_list`, on all processes.
# * `dist.barrier(group)`: block all processes in *group* until each one has entered this function.

# %%
""" All-Reduce example. """
def run_collective_all_reduce(rank, size):
    """ Simple collective communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])

# %% [markdown] 
# ## Distributed Training
# A didactic example replicate the functionality of DistributedDataParallel.
class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def partition_dataset():
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        './data/MNIST',
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=int(bsz), shuffle=True)
    return train_set, bsz

def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def run_demo(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)

# %%
if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run_demo))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()