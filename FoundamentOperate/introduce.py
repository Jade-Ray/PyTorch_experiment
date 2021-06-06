# %%[markdown]
# Introduce common torch tensor operation
from __future__ import print_function
import torch
import numpy as np

x = torch.empty(5, 3)
print(f"Empty 5*3 Tensor: {x}")

x = torch.rand(5, 3)
print(f"Random 5*3 Tensor: {x}")

x = torch.zeros(5, 3, dtype=torch.long)
print(f"Zeros 5*3 Tensor: {x}")

x = torch.tensor([5.5, 3])
print(f"Defined Tensor: {x}")

x = x.new_ones(5, 3, dtype=torch.double)
print(f"Ones 5*3 Tensor: {x}")
x = torch.rand_like(x, dtype=torch.float)
print(f"Random like Ones (5*3) Tensor: {x}")

print(f"Tensor Size : {x.size()}")

# %%[markdown]
# Introduce add operate calculation with different way.
y = torch.rand(5, 3)
print(f"Using + operate -> {x + y}")
print(f"Using add operate -> {torch.add(x, y)}")

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(f"Using out in add operate -> {result}")

y.add_(x)
print(f"Using inline add_ operate -> {y}")

# %%[markdown]
# Introduce slice and transform about tensor.
print(f"Slice operate -> {x[:, 1]}")

x= torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(f"Origin size -> {x.size()} \nTransform to 1 * 16 -> {y.size()} \nTransform to auto*8 -> {z.size()}")

# %%[markdown]
# Introduce get data from tensor.
x = torch.rand(1)
print(f"Tensor -> {x}")
print(f"Tensor data -> {x.item()}")

# %%[markdown]
# Intorduce Numpy <==> Tensor
a = torch.ones(5)
print(f"Tensor -> {a}")
b = a.numpy()
print(f"To Numpy -> {b}")

a.add_(1)
print(f"Tensor data add 1 -> {a}")
print(f"Numpy data refe tensor data -> {b}")

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(f"Numpy data add 1 -> {a}")
print(f"Tensor data refe numpy data -> {b}")

# %%[markdown]
# CUDA Tensors
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(f"tensor with cuda -> {z}")
    print(f'Conver to cpu norm tensor -> {z.to("cpu", torch.double)}')
# %%
