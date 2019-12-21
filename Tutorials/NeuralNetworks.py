import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # 6, 30, 30
        self.conv2 = nn.Conv2d(6, 16, 3) # 16, 13, 13

        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 6, 15, 15
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # 16, 6, 6
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        print(x.size())
        x = self.fc3(3)
        return x 

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            print(s)
            num_features *= s
        print(num_features)
        return num_features

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
for p in params:
    print(p.size())

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)