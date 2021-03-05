# %% [markdown]
# # pytorch TextDataset setting👨‍🦲
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
# %% [markdown]
# ## loader dataset
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"), 
                            init_token='<sos>', 
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT, root="../../Data")
# %%[Markdown]
# ## 
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    nbatch = data.size(0)
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)