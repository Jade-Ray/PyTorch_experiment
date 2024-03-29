# %% [markdown]
# # End-to-End Object Detection with Transformers
# *[refer from offical git repo](https://github.com/facebookresearch/detr)*

# %%
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from torchvision.models import resnet50 
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# 💠Load COCO Dataset and generate train dataloader


# %% [markdown]
# 💠Generate DETR Net 

# %%
class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positionall encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for prediction non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positonal encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x) # [n, 2048, 7, 7]

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x) # [n, 256, 7, 7]

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1), # [7, 7, 128]
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1), # [7, 7, 128]
        ], dim=-1).flatten(0, 1).unsqueeze(1) # [7, 7, 256] => [49, 256] => [49, 1, 256]

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1), 
                             self.query_pos.unsqueeze(1)).transpose(0, 1)  # src: [49, 1, 256], tgt: [100, 1, 256], out: [100, 1, 256] => [1, 100, 256]

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}

detr = DETRdemo(num_classes=91)
state_dict = torch.load('data/detr_demo-da2a99e9.pth', map_location='cpu')
detr.load_state_dict(state_dict)
detr.to(device).eval()

# %% [markdown]
# ## Computing predictions with DETR
# The pre-trained DETR model trained on the 80 COCO classes, with class indices ranging from 1 to 90.

# %%
# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# %% [markdown]
# DETR uses standard ImageNet normalization, and output boxes in relative image coordinates in [Xcenter, Ycenter, W, H] format. Because the coordinates are relative to the imagae dimension and lies between [0. 1], we convert predictions to absolute image coordinates.

# %%
# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800), 
    T.ToTensor(), 
    T.Normalize([0.485, 0.4566, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b

# %% [markdown]
# ## Using DETR detect image and visualize the predictions

# %%
def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img.to(device))

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1] # [100, 91]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep].cpu(), bboxes_scaled.cpu()

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

im = Image.open('data/00002.jpg')

scores, boxes = detect(im, detr, transform)

plot_results(im, scores, boxes)

# %% [markdown]
# ## Detection-Visualize encoder-decoder multi-head attention weights
#
# using hooks to extract attention weights (averaged over all heads) from the transformer.

# %%
# use lists to store the outputs via up-values
conv_features, enc_atten_weights, dec_atten_weights = [], [], []

hooks = [
    detr.backbone.layer4.register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    ),
    detr.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_atten_weights.append(output[1])
    ),
    detr.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_atten_weights.append(output[1])
    ),
]

# propagate through the model
with torch.no_grad():
    outputs = detr(transform(im).unsqueeze(0).to(device))
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1].cpu() # [100, 91]
keep = (probas.max(-1).values > 0.7)
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size).cpu()

for hook in hooks:
    hook.remove()

# don't need the list anymore
conv_features = conv_features[0].cpu()
enc_atten_weights = enc_atten_weights[0].cpu()
dec_atten_weights = dec_atten_weights[0].cpu()

# %%
# get the feature map shape
h, w = conv_features.shape[-2:]

fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
colors = COLORS * 100
for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    ax = ax_i[0]
    ax.imshow(dec_atten_weights[0, idx].view(h, w))
    ax.axis('off')
    ax.set_title(f'query id: {idx.item()}')
    ax = ax_i[1]
    ax.imshow(im)
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='blue', linewidth=3))
    ax.axis('off')
    ax.set_title(CLASSES[probas[idx].argmax()])
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize encoder self-attention weights
# the self-attention is square matrix of size [H * W, H * W], reshape it so that it has a more interpretable representation of [H, W, H, W]

# %%
# output of the CNN
f_map = conv_features
print('Encoder attention:               ', enc_atten_weights[0].shape)
print('Feature map:                     ', f_map.shape)

# %%
# get the HxW shape of the feature maps of the CNN
shape = f_map.shape[-2:]
# and reshape the self-attention to a more interpretable shape
sattn = enc_atten_weights[0].reshape(shape + shape)
print("Reshaped self-attention:", sattn.shape)

# %%
# downsampling factory for the CNN, is 32 for DETR and 16 for DETR DC5
fact = 32

# Let's select 4 reference points for visualization
idxs = [(150, 160), (300, 250), (200, 600), (440, 800),]

# here we create the canvas
fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
# and we add one plot per reference point
gs = fig.add_gridspec(2, 4)
axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[0, -1]),
    fig.add_subplot(gs[1, -1]),
]

# for each one of the reference points, Let's plot the self-attention
# for that point
for idx_o, ax in zip(idxs, axs):
    idx = (idx_o[0] // fact, idx_o[1] // fact)
    ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
    ax.axis('off')
    ax.set_title(f'self-attention{idx_o}')

# and now let's add the central image, with the reference points as red circles
fcenter_ax = fig.add_subplot(gs[:, 1:-1])
fcenter_ax.imshow(im)
for (y, x) in idxs:
    scale = im.height / transform(im).unsqueeze(0).shape[-2]
    x = ((x // fact) + 0.5) * fact
    y = ((y // fact) + 0.5) * fact
    fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
    fcenter_ax.axis('off')

plt.show()

# %%
