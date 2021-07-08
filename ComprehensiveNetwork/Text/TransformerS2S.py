# %% [markdown]
# # 0️⃣Define the model
# `TransformerEncoder` is a stack of N encoder layers
# `TransformerEncoderLayer` is made up of self-attn and feedforward network.

# import logging
# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
# logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    # Return mask => [[0.0, -inf, -inf],
    #                 [0.0,  0.0, -inf],
    #                 [0.0,  0.0,  0.0]]
    def generate_square_subsequent_mask(self, sz):
        # triu return the upper triangular part of matrix(2D), and transpose to lower triangular with bool type.
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # [seq_len, batch_size], [seq_len, seq_len]
    def forward(self, src, src_mask):
        # [seq_len, batch_size] => [seq_len, batch_size, ninp]
        src = self.encoder(src) * math.sqrt(self.ninp)
        # [seq_len, batch_size, ninp](add pos_encoder)
        src = self.pos_encoder(src)
        # [seq_len, batch_size, ninp](after scaled Dot-Product Multi-Head Attention)
        output = self.transformer_encoder(src, src_mask)
        # [seq_len, batch_size, ninp] => [seq_len, batch_size, ntoken]
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # [max_len] => [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # [(d_model + 1) // 2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # [max_len, i^2] = sin(position * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        # [max_len, i^2 + 1] = cos(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [1, max_len, d_model] => [max_len, 1, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x += [x.size(0), 1, d_model], x内的每个元素对应加
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# %% [markdown]
# # 1️⃣Load and batch data
import os
import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, vocab

train_iter = WikiText2('data', split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def data_process(raw_text_iter):
    # tokenizer to seg text, and vocab to trans to num
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    # discard 0 element text and cat them. numel func to vector element num.
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = WikiText2(root='data',)
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data:torch.Tensor, bsz:int):
    # Divide the dataset into bsz parts
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    # [data.size(0)] => [nbatch * bsz] 
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    # [nbatch * bsz] => [bsz, nbatch] => [nbatch, bsz]
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, batch_size)
test_data = batchify(test_data, batch_size)

# %% [markdown]
# 💠***Functions to generate input and target sequence***
# get_batch() function generates the input and target sequence for the transformer model. It subdivides the source data into chunks of length bptt. 
bptt = 35
def get_batch(source:torch.Tensor, i):
    seq_len = min(bptt, len(source) - 1 - i)
    # data => [seq_len, batch_size]
    data = source[i:i+seq_len]
    # target => [seq_len, batch_size] => [seq_len * batch_size]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

# %% [markdown]
# 💠***Initiate an instance***
ntokens = len(vocab) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

# %% [markdown]
# # 2️⃣Run the model
#
# CrossEntropyLoss is applied to track the loss and SGD implements stochastic gradient descent method as the optimizer. The initial learning rate is set to 5.0. StepLR is applied to adjust the learn rate through epochs. During the training, we use nn.utils.clip_grad_norm_ function to scale all the gradient together to prevent exploding.
import time

criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(epoch:int):
    model.train() # Turn on the train mode
    total_loss = 0
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)
            ))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

best_val_loss = float("inf")
epochs = 3 # the number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(epoch)
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
        epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
    ))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

# %% [markdown]
# 💠***Evaluate the model with the test dataset***
test_loss = evaluate(best_model, test_data)
print("=" * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print("=" * 89)
# %%
