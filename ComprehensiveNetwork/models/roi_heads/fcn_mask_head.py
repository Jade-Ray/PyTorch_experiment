import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ComprehensiveNetwork.models.cnn import ConvModule


class FCNMaskHead(nn.Module):
    
    def __init__(self, num_convs=4, in_channels=256, 
                 conv_kernel_size=3, conv_out_channels=256, num_classes=80):
        super(FCNMaskHead, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (self.in_channels if i==0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(ConvModule(in_channels, self.conv_out_channels,
                                         self.conv_kernel_size, padding=padding))
        upsample_in_channels = (self.conv_out_channels if self.num_convs > 0 else in_channels)
        self.upsample = nn.ConvTranspose2d(in_channels=upsample_in_channels, 
                                           out_channels=self.conv_out_channels,
                                           kernel_size=2, stride=2)
        self.conv_logits = nn.Conv2d(self.conv_out_channels, self.num_classes, 1)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.relu(self.upsample(x))
        mask_pred = self.conv_logits(x)
        return mask_pred
        
