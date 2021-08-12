# %% [markdown]
# # Applicate SimCLR with PyTorch referred [official github repo](https://github.com/sthalles/SimCLR)

# %%
import torch
import torch.backends.cudnn as cudnn
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import models, datasets, transforms
from torchvision.transforms.transforms import GaussianBlur

import os
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
    # imporve the efficient of convolution operation only used in invariable input size net
    cudnn.deterministic = True 
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

from torch.utils.tensorboard import SummaryWriter

LOG_DIR = 'runs/SimCLR_experiment_1'
writer = SummaryWriter(LOG_DIR)

# %% [markdown]
# 💠Load and generate dataset

# %%
ROOT_FOLDER = 'data'
DATASET_NAME = 'stl10'
VIEW_NUM = 2 # Number of views for contrastive learning training
BATCH_SIZE = 256
NUM_WORKERS = 0

def get_simclr_pipeline_transform(size, s=1):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size), 
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8), 
                                          transforms.RandomGrayscale(p=0.2), 
                                          GaussianBlur(kernel_size=int(0.1 * size)), 
                                          transforms.ToTensor()])
    return data_transforms

class ContrastiveLearningViewGenerator(object):
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views        
    
    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

def get_dataset():
    vailid_datasets = {'cifar10': lambda: datasets.CIFAR10(root=ROOT_FOLDER, train=True, 
                                                       transform=ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(32), VIEW_NUM), 
                                                       download=True), 
                        'stl10': lambda: datasets.STL10(root=ROOT_FOLDER, split='unlabeled', 
                                                    transform=ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(96), VIEW_NUM), 
                                                    download=True)}
    try:
        dataset_fn = vailid_datasets[DATASET_NAME]
    except KeyError:
        raise Exception("dataset now only supported cifa10 and stl10")
    return dataset_fn()

train_dataset = get_dataset()

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last = True)

# %% [markdown]
# *imshow dataset and randomly DA images*

# %%
import matplotlib.pyplot as plt
import numpy as np
import torchvision

print(f'imshow first {DATASET_NAME} dataset origin image')
plt.imshow(np.transpose(train_dataset.data[1], (1, 2, 0)))
plt.show()

# %%
print(f'imshow {DATASET_NAME} dataset DA compared images')
dataiter = iter(train_loader)
images, _ = dataiter.next()
compared_images = torch.cat((images[0][:4], images[1][:4]), 0)
img_grid = torchvision.utils.make_grid(compared_images, nrow=4)
plt.imshow(np.transpose(img_grid, (1, 2, 0)))
plt.show()
writer.add_image('data_augment_compared_images', img_grid)

# %% [markdown]
# 💠Generate SimCLR Net model

# %%
BASE_MODEL = 'resnet18'
OUT_DIM = 128
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-4

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim), 
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}
        try:
            self.backbone = self.resnet_dict[base_model]
        except KeyError:
            raise Exception("base_model now only supported resnet18 and resnet50")
        
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(BATCH_SIZE) for i in range(VIEW_NUM)], dim=0) # [BS * VN]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() # [BS * VN, BS * VN] eye matrix
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positive
        positive = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positive, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / 0.07
        return logits, labels

model = ResNetSimCLR(base_model=BASE_MODEL, out_dim=OUT_DIM).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

criterion = torch.nn.CrossEntropyLoss().to(device)

# %% [markdown]
# 💠Training model

# %%
EPOCHS = 200
LOG_EVERY_STEP = 5

def accuracy(output: torch.Tensor, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train():
    scaler = GradScaler() # automatic mixed precision(AMP)

    n_iter = 0
    print(f"Start SimCLR training for {EPOCHS} epochs.")
    print(f"Training with gpu: {torch.cuda.is_available()}.")

    for epoch_counter in range(EPOCHS):
        for images, _ in tqdm(train_loader):
            images = torch.cat(images, dim=0)

            images = images.to(device)

            with autocast():
                features = model(images)
                logits, labels = model.info_nce_loss(features)
                loss = criterion(logits, labels)

            optimizer.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if n_iter % LOG_EVERY_STEP == 0:
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                writer.add_scalar('loss', loss, global_step=n_iter)
                writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step=n_iter)

            n_iter += 1

        # warmup for the first 10 epochs
        if epoch_counter >= 10:
            scheduler.step()
        print(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

    print("Training has finished.")
    # save model checkpoints
    checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(EPOCHS)
    torch.save({
        'epoch': EPOCHS,
        'arch': BASE_MODEL,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(LOG_DIR, checkpoint_name))
    print(f"Model checkpoint and metadata has beeen saved at {LOG_DIR}.")

# %%
train()