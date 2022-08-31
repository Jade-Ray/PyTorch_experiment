import torch
import torch.nn as nn

from .convfc_bbox_head import Shared2FCBBoxHead
from .fcn_mask_head import FCNMaskHead


class StandardRoiHead(nn.Module):
    """Simplest base roi head including one bbox head and one mask head."""
    
    def __init__(self, bbox_head=None, mask_head=None):
        super(StandardRoiHead, self).__init__()
        self.bbox_head = Shared2FCBBoxHead(**bbox_head)
        self.mask_head = FCNMaskHead(**mask_head)
