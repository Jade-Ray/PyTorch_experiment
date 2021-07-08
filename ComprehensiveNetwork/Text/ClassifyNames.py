# %% [markdown]
# # 1️⃣Preparing the Data
# ***Classifying names with a character-level RNN***. 
# Not using many of convenience functions of torchtext.

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

from torch import tensor

def findFiles(path):
    return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,:'"
n_letters = len(all_letters)
# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
# %% [markdown]
# 💠***test for preparing process***

print(category_lines['Polish'][:5])

# %% [markdown]
# # 2️⃣Turning Names into Tensors
# Use one-hot vector of size `<1 x n_letters>`

import torch

# Find letter index from all_letters. e.g. "a" = 0
def letter2Index(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter2Tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter2Index(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>, or an array of one-hot letter vectors
def line2Tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter2Index(letter)] = 1
    return tensor

# %% [markdown]
# 💠***test for turing tensor process***
print(letter2Tensor('J'))
print(line2Tensor('Jones').size())

# %% [markdown]
# # 3️⃣Creating the Network

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # <1 x n_letters> concatenate <1 x n_hidden> equal <1 x (n_letters + n_hidden)>
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# %% [markdown]
# 💠***test for Network***
input = letter2Tensor('A')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)
print(output)

input2 = line2Tensor("Albert")
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input2[0], hidden)
print(output)

# %% [markdown]
# # 4️⃣Training
# 💠***preparing for training***

def categoryFromOutput(output):
    # use Tensor.topk to get the index of the greatest value
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))

import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line2Tensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category = ', category, ' / line = ', line)

# %% [markdown]
# 💠***training the network***
criterion = nn.NLLLoss()
learning_rate = 0.005

# each loop of training will:
# - create input and target tensors
# - create a zeroed initial hidden state
# - read each lettere in and keep hidden state for next letter
# - compare final output to target
# - back_propagate
# - return the output and loss
def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    
    return output, loss.item()

import math
import time

n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

start = time.time()

print("\nBegin training......\n")
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i  = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print("{} {}% ({}) {:.4f} {} / {} {}".format(iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# %% [markdown]
# 💠  ***ploting the result***
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

# %% [markdown]
# # 5️⃣Evaluating and Results

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories) 
n_confusion = 10000

# Just return an output given a ine
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# set up plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# force labeel at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

# %% [markdown]
# 💠***Running on user input***
def predict(input_line, n_predictions=3):
    print('> %s' % input_line)
    with torch.no_grad():
        output = evaluate(line2Tensor(input_line))

        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print("({:.2f}) {}".format(value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('LeiQi')
predict('Satoshi')

# %%
