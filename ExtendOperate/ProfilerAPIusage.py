# %% [markdown]
# # `PyTorch profiler API` is useful to identify the itme and memory costs of various PyTorch operations in your code.

import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler


# %% [markdown]
# 💠Build a custom module that performs two sub-tasks:
# - a linear transformation on the input, and
# - use the transformation result to get indices on a mask tensor.
#
# ---
# wrap the code for each sub-task in separate labelled context managers using `profiler.record_function("label")`. Note that using Profiler incurs some overhead, and is best used only for investigating code. Remember to remove it if you are benchmarking runtimes.

class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)
        
        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean().item()
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
            hi_idx = torch.from_numpy(hi_idx).cuda()

        return out, hi_idx
# %% [markdown]
# 💠Profile the forward pass
#
# ---
# we wrap the forward pass of our module in the `profiler.profile` context manager. The `with_stack=True` parameter appends the file and line number of the operation in the trace.

model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.double).cuda()

model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

# %% [markdown]
# 💠Print profiler results
#
# ---
# `profiler.key_averages` aggregates the results by operator name, and optionally by input shapes and/or stack trace events

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

# %% [markdown]
# 💠We can see that line 12 consumes 953.67Mb. so we can reduce memory footprint by casting it to `torch.float`.

model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()

model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

# %% [markdown]
# 💠The `aten::copy_` operator in `forward(12)` copies mask to CPU so expensive! We use torch function `nonzero()` instead to improve time performance
class MyModule2(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule2, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)
        
        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean()
            hi_idx = (mask > threshold).nonzero(as_tuple=True)

        return out, hi_idx

model = MyModule2(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()

model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

# %%
