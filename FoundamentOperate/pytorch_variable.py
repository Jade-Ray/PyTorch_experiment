# %% [markdown]
# Variable opration introduction
import torch
from torch.autograd import Variable # 加载torch 中 Variable 模块

# %% [markdown]
# 先生鸡蛋
tensor = torch.FloatTensor([[1, 2], [3, 4]])
# 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度
variable = Variable(tensor, requires_grad=True)

print(tensor)
print(variable)

# %% [markdown]
# 添加计算操作 
# t_out = 1/4 * sum(tensor * tensor)
# v_out = 1/4 * sum(variable * varibale)
t_out = torch.mean(tensor * tensor)
v_out = torch.mean(variable * variable)

print(t_out)
print(v_out)

# %% [markdown]
# 使用反向传播自动计算梯度
v_out.backward()    # 模拟 v_out 的误差反向传递
# 下面两步看不懂没关系, 只要知道 Variable 是计算图的一部分, 可以用来传递误差就好.
# v_out = 1/4 * sum(variable*variable) 这是计算图中的 v_out 计算步骤
# 针对于 v_out 的梯度就是, d(v_out)/d(variable) = 1/4 * 2 * variable = variable/2
print(variable.grad)    # 初始 Variable 的梯度
print(variable.data)
print(variable.data.numpy())

# %%
