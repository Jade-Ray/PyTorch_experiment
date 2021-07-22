# %% [markdown]
# # 0️⃣Access to the raw dataset iterators
# Build the dataset for the text classification analysis using the torchtext library
# ---
# - `AG_NEWS` dataset iterators yield the raw data as a tuple of label and text
# - `AG_NEWS` dataset has four labels
#   - 1 : World
#   - 2 : Sports
#   - 3 : Business
#   - 4 : Sci/Tec
import torch
from torchtext.datasets import AG_NEWS
train_iter = AG_NEWS(root='data', split='train')
print(next(train_iter))
print(next(train_iter))
print(next(train_iter))

# %% [markdown]
# # 1️⃣Prepare data processing piplines
# ---
# - very basic components of the torchtext including vocab, word vectors, tokenizer
# - build a vocabulary with the raw training dataset through factory function `build_vocab_from_iterator` which accepts iterator that yield list or iterator of tokens. And users can also pass any special symbols to be added to the vocabulary
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(root='data', split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

print(vocab(['here', 'is', 'an', 'example']))

# prepare the text processing pipeline with the tokenizer and vocabulary
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

print(text_pipeline('here is the an example.'))
print(label_pipeline('10'))

# %% [markdown]
# # 2️⃣Generate data batch and iterator
# ---
# - `DataLoader` works with a map-style dataset that inplements the `getitem()` and `len()` protocols, and it also works with an iterable dataset
# - before sending to the model, `collate_fn` function works on a batch of samples generated from `DataLoader`, which processes input accorrding to the data processing pipelines declaared previously 
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    # cumsum return the cumulative sum of the elements along a given axis
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

train_iter = AG_NEWS(root='data', split='train')
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

# %% [markdown]
# # 3️⃣Define the model
# the model is composed of the `nn.EmbeddingBag` layer plus a linear layer for the classification purpose
# ---
# - `nn.EmbeddingBag` with the default mode of "mean" computes the mean value of a "bag" of embeddings. Although the text entries here have different lengths, which module requires no padding here since the text lengths arre saved in offsets.
# - `nn.EmbeddingBag` can enhance the performance and memory efficiency to process a sequence of tensors since which accumulates the average across the embeddings on the fly.
# - `mean` mode equivalent to `Embbedding` followed by `torch.mean(dim=0)`
from torch import nn

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# Build a model with embedding dimension of 64, and vocab size is equal to the length of the vocabulary instance, and the number of classes is equal to the number of labels
train_iter = AG_NEWS(root='data', split='train')
num_class = len(set([label for (label, text) in train_iter ]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

# %% [markdown]
# # 4️⃣Define functions to train and evaluate
import time

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offset) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, offset)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print("| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}".format(epoch, idx, len(dataloader), total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0 

    with torch.no_grad():
        for idx, (label, text, offset) in enumerate(dataloader):
            predited_label = model(text, offset)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

# %% [markdown]
# 💠***split the dataset and run the model***
# ---
# - split `AG_NEWS` training dataset into train/valid with ratio of 0.95 and 0.05
# - `CrossEntropyLoss` criterion combines `LogSoftmax()` and `NLLLoss()` in a single class, which is useful in classification problem
# - `SGD` implements stochastic gradient descent method as the optimizer
# - `StepLR` is used to adjust the learning rate through epochs
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
# Hyperparameters
EPOCHS = 10 # epoch
LR = 5 # learning rate
BATCH_SIZE = 64 # batch size for training

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_acc = None
train_iter, test_iter = AG_NEWS(root='data')
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_acc is not None and total_acc > accu_val:
        scheduler.step()
    else:
        total_acc = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid accuracy {:8.3f} '.format(epoch, time.time() - epoch_start_time, accu_val))
    print('-' * 59)

# %% [markdown]
# # 5️⃣Evaluate the model with test dataset
print('Checking the results of test dataset.')
acc_test = evaluate(test_dataloader)
print('test accuracy {:8.3f}'.format(acc_test))

# %% [markdown]
# 💠***Test on a random news***
ag_news_label = {1: 'World', 2: "Sports", 3: "Business", 4:"Sci/Tec"}

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

model = model.to('cpu')
print("This is a %s news" %ag_news_label[predict(ex_text_str, text_pipeline)])

# %%
