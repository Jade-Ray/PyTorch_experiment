# %% [markdown]
# # 1️⃣Preparing the Data
# ***Generating names from languages with a character-level RNN***. 
# hand-crafting a samll RNN with a few linear layers.
# *One to sequence*

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

from torch import tensor

all_letters = string.ascii_letters + " .,:'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

category_lines = {}
all_categories = []
for filename in findFiles('data/names/*txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

print('# categories: ', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))

# %% [markdown]
# # 2️⃣Greating the Network
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(output_size + hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((output, hidden), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_letters)

# %% [markdown]
# 💠***LSTM model network***
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(n_categories + input_size, hidden_size)
        self.o2o = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, categroy, input, hidden):
        input_combined = torch.cat((categroy, input), 1).unsqueeze(0)
        output, hidden = self.lstm(input_combined, hidden)
        output = self.o2o(output)
        output = self.dropout(output)
        output = self.softmax(output[0])
        return output, hidden

    def initHidden(self):
        h0 = torch.zeros(1, 1, self.hidden_size)
        c0 = torch.zeros(1, 1, self.hidden_size)
        return (h0, c0)

lstm = LSTM(n_letters, n_hidden, n_letters)

# %% [markdown]
# 💠***GRU model network***
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(n_categories + input_size, hidden_size)
        self.o2o = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, categroy, input, hidden):
        input_combined = torch.cat((categroy, input), 1).unsqueeze(0)
        output, hidden = self.gru(input_combined, hidden)
        output = self.o2o(output)
        output = self.dropout(output)
        output = self.softmax(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

gru = GRU(n_letters, n_hidden, n_letters)

# %% [markdown]
# # 3️⃣Training
# 💠***preparing for training***
import random

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

# %% [markdown]
# 💠***training the Network***
criterion = nn.NLLLoss()
learning_rate = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor, net=rnn):
    target_line_tensor.unsqueeze_(-1)
    hidden = net.initHidden()
    net.zero_grad()
    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = net(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l
    
    loss.backward()

    for p in net.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)

import math
import time

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # reset every plot_every iter

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample(), net=rnn)
    total_loss += loss

    if iter % print_every == 0:
        print("{} ({} {}%) {:.4f}".format(timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

# %% [markdown]
# 💠***ploting the losses***
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

# %% [markdown]
# # 4️⃣Sampling the Network
max_length = 20

def sample(category, start_letter='A', net=rnn):
    with torch.no_grad():
        category_tensor = categoryTensor(category)
        input_tensor = inputTensor(start_letter)
        hidden = net.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = net(category_tensor, input_tensor[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input_tensor = inputTensor(letter)

    return output_name

def samples(category, start_letters='ABC'):
    print("\nGenerating names from {}:".format(category))
    for start_letter in start_letters:
        print(sample(category, start_letter))

samples('Russian', 'RUS')
samples('German', 'GER')
samples('Spanish', 'SPA')
samples('Chinese', 'CHI')

# %%
