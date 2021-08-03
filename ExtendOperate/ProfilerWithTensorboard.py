# %% [markdown]
# # Use TensorBoard plugin with PyTorch Profiler to ditect performance bottlenecks of the model.
#
# ---
# - Prepare data and model
# - Use profiler to record execution events
# - Run the profiler
# - Use TensorBoard to view results and analyze model performance
# - Imporve performance with the help of profiler
# - Analyze performance with other advanced features

import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T

transform = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='/data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

device = torch.device("cuda:0")
model = torchvision.models.resnet18(pretrained=True).cuda(device)
criterion = torch.nn.CrossEntropyLoss().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()

def train(data):
    inputs, labels = data[0].to(device), data[1].to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %% [markdown]
# The profiler is enabled through the context manager and accepts several parameters
# ---
# - `scheduler`: callable that takes step as a single parameter and returns the profiler action to perform at each step. During `wait` steps, the profiler is disabled. During `warmup` steps, the profiler starts tracing but the results are discarded.
# - `on_trace_ready`: callable that is called at the end of each cycle
# - `record_shapes`: whether to record shapes of the operator inputs
# - `profile_memory`: Track tensor memory allocation/deallocation
# - `with_stack`: Record source information(file and line number) for the ops

with torch.profiler.profile(
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2), 
    on_trace_ready = torch.profiler.tensorboard_trace_handler('log/resnet18'),
    record_shapes=True, 
    with_stack=True
) as prof:
    for step, batch_data in enumerate(train_loader):
        if step >= (1 + 1 + 3) * 2:
            break
        train(batch_data)
        prof.step() # Need to call this at the end of each step to notify profiler of steps' boundary

# %%
