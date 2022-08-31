# %% [markdown]
# # Swin Transformer Object Detection with Mask RCNN
# *[refer from offical git repo](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)*

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

# %%
import torch
import torch.nn as nn

from ComprehensiveNetwork.models.backbones import SwinTransformer
from ComprehensiveNetwork.models.necks import FPN
from ComprehensiveNetwork.models.dense_heads import RPNHead 
from ComprehensiveNetwork.models.roi_heads import StandardRoiHead


class MaskRCNN(nn.Module):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""
    def __init__(self):
        super(MaskRCNN, self).__init__()
        self.backbone = SwinTransformer(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], 
                                        window_size=7, ape=False, drop_path_rate=0.1, patch_norm=True)
        self.neck = FPN(in_channels=[96, 192, 384, 768], out_channels=256, num_outs=5)
        self.rpn_head = RPNHead(in_channels=256, feat_channels=256, 
                                anchor_generator=dict(type='AnchorGenerator', scales=[8], 
                                                      ratios=[0.5, 1.0, 2.0], 
                                                      strides=[4, 8, 26, 32, 64]))
        self.roi_head = StandardRoiHead(
            bbox_head=dict(in_channels=256, fc_out_channels=1024, roi_feat_size=7, num_classes=80),
            mask_head=dict(num_convs=4, in_channels=256, conv_out_channels=256, num_classes=80))
    
    def forward(self, img, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        # Directly extract features from the backbone+neck.
        # 4 levle output:
        # - B C  H/4  W/4
        # - B 2C H/8  W/8
        # - B 4C H/16 W/16
        # - B 8C H/32 W/32
        x = self.backbone(img)
        x = self.neck(x) # to 256 dim
        
        rpn_outs = self.rpn_head()
        proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_metas)
        
        pass


# %% [markdown]
# - Building model with Default Mask R-CNN Swin-T: `C=96, layer numbers={2, 2, 6, 2}`

# %%
model = MaskRCNN()

print(str(model))

model.cuda()

checkpoint = torch.load('../../Data/PretrainedModel/mask_rcnn_swin_tiny_patch4_window7_1x.pth', map_location='cpu')
meta = checkpoint['meta']
msg = model.load_state_dict(checkpoint['state_dict'])
print(msg)

# %% [markdown]
# - Predicting model
# just in one batch simply    

# %%
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = Image.open('../../Data/Images/dog_bike_car.jpg')

coco_labels = meta['CLASSES']

with torch.no_grad():
    model.eval()
    input = transform(image.convert("RGB")).unsqueeze(0).cuda(non_blocking=True)


# %%
