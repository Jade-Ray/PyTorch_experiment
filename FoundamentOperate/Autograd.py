# %%[markdown]
# Introducation torch autograd operation.
import torch

# %%[markdown]
# Set and check auto grad of torch
a = torch.rand(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# %%[markdown]
# Print grad_fn of torch
#
# `z = 3 * (x + 2)^2`
x = torch.ones(2, 2, requires_grad=True)
print(f"x -> {x}")

y = x + 2
print(f"y = x + 2 -> {y}")

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(f"z = 3 * y^2 -> {z} \nout = mean(z) -> {out}")

# %%[markdown]
# Calculate grad and print 
# 
# `dout/dx = d(1/4 * sum(3 * (x + 2)^2)) / dx => 3/2 * (x +2)`
out.backward()
print(x.grad)

# %%[markdown]
# Complexity calculation
x = torch.rand(3, requires_grad=True)

y = x * 2
repeat_num = 0
while y.data.norm() < 1000:
    y = y * 2
    repeat_num += 1

print(f"x -> {x} \ny = x^{repeat_num * 2} -> {y}")

# convert tensor y to scalar, only scalar can calcu grad.
# sum(v dot y)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(f"grandient v -> {v}")
print(f"d(sum(v dot x^{repeat_num * 2})) / dx -> {x.grad}")

# %%[markdown]
# `no_grad` func can cancel auto grad.
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

# %%[markdown]
# `detach` func create tensor without auto grad, although with equal data.
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y), x.eq(y).all())

# %%
