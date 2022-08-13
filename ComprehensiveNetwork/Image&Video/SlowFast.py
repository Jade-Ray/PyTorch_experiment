# %% [markdown]
# # SlowFast Networks for Video Recognition
# *[refer from offical git repo](https://github.com/facebookresearch/SlowFast)*

# %%
import torch
import torch.nn as nn

from torchinfo import summary


#-----------------------------------------------------------------
# Stem module
# ----------------------------------------------------------------
class ResNetBasicStem(nn.Module):
    """
    ResNe(X)t 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """
    
    def __init__(self, dim_in, dim_out, kernel, stride, padding, inplace_relu=True, 
                 eps=1e-5, bn_mmt=0.1, norm_module=nn.BatchNorm3d,):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module)
    
    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv = nn.Conv3d(dim_in, dim_out, self.kernel, 
                              stride=self.stride, padding=self.padding, bias=False)
        self.bn = norm_module(num_features=dim_out, eps=self.eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(self.inplace_relu)
        self.pool_layer = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool_layer(x)
        return x


class VideoModelStem(nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """
    
    def __init__(self, dim_in, dim_out, kernel, stride, padding, inplace_relu=True, 
                 eps=1e-5, bn_mmt=0.1, norm_module=nn.BatchNorm3d, stem_func_name="basic_stem"):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, Slow
        and etc), list size of 2 for two pathway models (SlowFast).

        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            stem_func_name (string): name of the the stem function applied on
                input to the network.
        """
        super(VideoModelStem, self).__init__()
        
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self._construct_stem(dim_in, dim_out, norm_module, stem_func_name)
        
    def _construct_stem(self, dim_in, dim_out, norm_module, stem_func_name):
        trans_func = ResNetBasicStem
        
        for pathway in range(len(dim_in)):
            stem = trans_func(dim_in[pathway], dim_out[pathway], 
                              self.kernel[pathway], self.stride[pathway], self.padding[pathway], 
                              self.inplace_relu, self.eps, self.bn_mmt, norm_module,)
            self.add_module(f'pathway{pathway}_stem', stem)
    
    def forward(self, x):
        assert (len(x) == self.num_pathways), f"Input tensor does not contain {self.num_pathways} pathway"
        # use a new list, don't modify in-place the x list, which is bad for activation checkpointing.
        y = []
        for pathway in range(len(x)):
            m = getattr(self, f"pathway{pathway}_stem")
            y.append(m(x[pathway]))
        return y


#-----------------------------------------------------------------
# Block module
# ----------------------------------------------------------------
def get_trans_func(name):
    """
    Retrieves the transformation module by name.
    """
    trans_funcs = {
        "bottleneck_transform": BottleneckTransform,
        "basic_transform": BasicTransform,
    }
    assert (
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]


class BasicTransform(nn.Module):
    """
    Basic transformation: Tx3x3, 1x3x3, where T is the size of temporal kernel.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner=None,
        num_groups=1,
        stride_1x1=None,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the first
                convolution in the basic block.
            stride (int): the stride of the bottleneck.
            dim_inner (None): the inner dimension would not be used in
                BasicTransform.
            num_groups (int): number of groups for the convolution. Number of
                group is always 1 for BasicTransform.
            stride_1x1 (None): stride_1x1 will not be used in BasicTransform.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(BasicTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(dim_in, dim_out, stride, dilation, norm_module)

    def _construct(self, dim_in, dim_out, stride, dilation, norm_module):
        # Tx3x3, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=[self.temp_kernel_size, 3, 3],
            stride=[1, stride, stride],
            padding=[int(self.temp_kernel_size // 2), 1, 1],
            bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)
        # 1x3x3, BN.
        self.b = nn.Conv3d(
            dim_out,
            dim_out,
            kernel_size=[1, 3, 3],
            stride=[1, 1, 1],
            padding=[0, dilation, dilation],
            dilation=[1, dilation, dilation],
            bias=False,
        )

        self.b.final_conv = True

        self.b_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )

        self.b_bn.transform_final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        x = self.b(x)
        x = self.b_bn(x)
        return x


class BottleneckTransform(nn.Module):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """
    
    def __init__(self, dim_in, dim_out, temp_kernel_size, stride, dim_inner, num_groups, 
                 stride_1x1=False, inplace_relu=True, eps=1e-5, bn_mmt=0.1, 
                 dilation=1, norm_module=nn.BatchNorm3d, block_idx=0,):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the first
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(BottleneckTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(dim_in, dim_out, stride, dim_inner, num_groups, dilation, norm_module,)
        
    def _construct(self, dim_in, dim_out, stride, dim_inner, num_groups, dilation, norm_module,):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)
        
        # Tx1x1, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x3x3, BN, ReLU.
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, dilation, dilation],
            groups=num_groups,
            bias=False,
            dilation=[1, dilation, dilation],
        )
        self.b_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN.
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c.final_conv = True

        self.c_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn.transform_final_bn = True
        
    def forward(self, x):
        # Explicitly forward every layer.
        # Branch2a.
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        # Branch2b.
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)

        # Branch2c
        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block.
    """
    
    def __init__(self, dim_in, dim_out, temp_kernel_size, stride, trans_func, dim_inner, 
                 num_groups=1, stride_1x1=False, inplace_relu=True, eps=1e-5, bn_mmt=0.1,
                 dilation=1, norm_module=nn.BatchNorm3d, block_idx=0, drop_connect_rate=0.0):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._drop_connect_rate = drop_connect_rate
        self._construct(dim_in, dim_out, temp_kernel_size, stride, trans_func, dim_inner, 
                        num_groups, stride_1x1, inplace_relu, dilation, norm_module, block_idx)
        
    def _construct(self, dim_in, dim_out, temp_kernel_size, stride, trans_func, dim_inner, 
                   num_groups, stride_1x1, inplace_relu, dilation, norm_module, block_idx):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=[1, stride, stride],
                                     padding=0, bias=False, dilation=1)
            self.branch1_bn = norm_module(num_features=dim_out, eps=self._eps, momentum=self._bn_mmt)
        self.branch2 = trans_func(dim_in, dim_out, temp_kernel_size, stride, dim_inner, num_groups,
                                  stride_1x1=stride_1x1, inplace_relu=inplace_relu, dilation=dilation,
                                  norm_module=norm_module, block_idx=block_idx)
        self.relu = nn.ReLU(self._inplace_relu)
            
    def forward(self, x):
        f_x = self.branch2(x)
        if self.training and self._drop_connect_rate > 0.0:
            f_x = drop_path(f_x, self._drop_connect_rate)
        if hasattr(self, "branch1"):
            x = self.branch1_bn(self.branch1(x)) + f_x
        else:
            x = x + f_x
        x = self.relu(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


#-----------------------------------------------------------------
# Layer module
# ----------------------------------------------------------------
class ResStage(nn.Module):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single pathway (C2D, I3D, Slow), and multi-pathway (SlowFast) cases.
        More details can be found here:

        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "SlowFast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """
    
    def __init__(self, dim_in, dim_out, stride, temp_kernel_sizes, num_blocks, dim_inner, num_groups, 
                 num_block_temp_kernel, nonlocal_inds, nonlocal_group, nonlocal_pool, dilation, 
                 instantiation="softmax", trans_func_name="bottleneck_transform", stride_1x1=False,
                 inplace_relu=True, norm_module=nn.BatchNorm3d, drop_connect_rate=0.0,):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_sizes (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_sizes
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_sizes to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            nonlocal_inds (list): If the tuple is empty, no nonlocal layer will
                be added. If the tuple is not empty, add nonlocal layers after
                the index-th block.
            dilation (list): size of dilation for each pathway.
            nonlocal_group (list): list of number of p nonlocal groups. Each
                number controls how to fold temporal dimension to batch
                dimension before applying nonlocal transformation.
                https://github.com/facebookresearch/video-nonlocal-net.
            instantiation (string): different instantiation for nonlocal layer.
                Supports two different instantiation method:
                    "dot_product": normalizing correlation matrix with L2.
                    "softmax": normalizing correlation matrix with Softmax.
            trans_func_name (string): name of the the transformation function apply
                on the network.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResStage, self).__init__()
        self.num_blocks = num_blocks
        self.nonlocal_group = nonlocal_group
        self._drop_connect_rate = drop_connect_rate
        self.temp_kernel_sizes = [
            (temp_kernel_sizes[i] * num_blocks[i])[: num_block_temp_kernel[i]]
            + [1] * (num_blocks[i] - num_block_temp_kernel[i])
            for i in range(len(temp_kernel_sizes))]
        self.num_pathways = len(self.num_blocks)
        self._construct(dim_in, dim_out, stride, dim_inner, num_groups, trans_func_name, stride_1x1,
                        inplace_relu, nonlocal_inds, nonlocal_pool, instantiation, dilation, 
                        norm_module)
        
    def _construct(self, dim_in, dim_out, stride, dim_inner, num_groups, trans_func_name, 
                   stride_1x1, inplace_relu, nonlocal_inds, nonlocal_pool, instantiation,
                   dilation, norm_module,):
        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                # Retrieve the transformation function.
                trans_func = get_trans_func(trans_func_name)
                # Construct the block.
                res_block = ResBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_out[pathway],
                    self.temp_kernel_sizes[pathway][i],
                    stride[pathway] if i == 0 else 1,
                    trans_func,
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway],
                    norm_module=norm_module,
                    block_idx=i,
                    drop_connect_rate=self._drop_connect_rate,
                )
                self.add_module("pathway{}_res{}".format(pathway, i), res_block)
    
    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            for i in range(self.num_blocks[pathway]):
                m = getattr(self, "pathway{}_res{}".format(pathway, i))
                x = m(x)
                if hasattr(self, "pathway{}_nonlocal{}".format(pathway, i)):
                    nln = getattr(
                        self, "pathway{}_nonlocal{}".format(pathway, i)
                    )
                    b, c, t, h, w = x.shape
                    if self.nonlocal_group[pathway] > 1:
                        # Fold temporal dimension into batch dimension.
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(
                            b * self.nonlocal_group[pathway],
                            t // self.nonlocal_group[pathway],
                            c,
                            h,
                            w,
                        )
                        x = x.permute(0, 2, 1, 3, 4)
                    x = nln(x)
                    if self.nonlocal_group[pathway] > 1:
                        # Fold back to temporal dimension.
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(b, t, c, h, w)
                        x = x.permute(0, 2, 1, 3, 4)
            output.append(x)

        return output
            


# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}
_MODEL_SCALE_RATIO = {18: (1, 2, 4, 8), 50: (4, 8, 16, 32), 101: (4, 8, 16, 32)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "slow_c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow_i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "slow_c2d": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "slow_i3d": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """
    
    def __init__(self, dim_in, fusion_conv_channel_ratio, fusion_kernel, alpha, 
                 eps=1e-5, bn_mmt=0.1, inplace_relu=True, norm_module=nn.BatchNorm3d,):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in, 
            dim_in * fusion_conv_channel_ratio, 
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1], 
            padding=[fusion_kernel // 2, 0, 0], 
            bias=False,)
        self.bn = norm_module(num_features=dim_in * fusion_conv_channel_ratio, 
                              eps=eps, momentum=bn_mmt,)
        self.relu = nn.ReLU(inplace_relu)
        
    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """
    
    def __init__(self, num_groups=1, width_per_group=64, beta_inv=8, alpha=8, 
                 fusion_conv_channel_ratio=2, fusion_kernel_size=7,
                 resnet_depth=18, model_arch='slowfast'):
        super(SlowFast, self).__init__()
        self.norm_module = nn.BatchNorm3d
        self.num_pathways = 2
        self._construct_network(num_groups, width_per_group, beta_inv, alpha,
                                fusion_conv_channel_ratio, fusion_kernel_size,
                                resnet_depth, model_arch)
        
    def _construct_network(self, num_groups, width_per_group, beta_inv, alpha,
                           fusion_conv_channel_ratio, fusion_kernel_size,
                           resnet_depth, model_arch):
        pool_size = _POOL1[model_arch]
        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[resnet_depth]
        (r2, r3, r4, r5) = _MODEL_SCALE_RATIO[resnet_depth]
        num_groups = num_groups
        width_per_group = width_per_group
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (beta_inv // fusion_conv_channel_ratio)
        temp_kernel = _TEMPORAL_KERNEL_BASIS[model_arch]
        
        self.stem = VideoModelStem(
            dim_in=[3, 3], 
            dim_out=[width_per_group, width_per_group // beta_inv], 
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]], 
            stride=[[1, 2, 2]] * 2,
            padding=[[temp_kernel[0][0][0] // 2, 3, 3],
                     [temp_kernel[0][1][0] // 2, 3, 3]],
            norm_module=self.norm_module)
        
        self.stem_fuse = FuseFastToSlow(
            width_per_group // beta_inv, 
            fusion_conv_channel_ratio, 
            fusion_kernel_size, 
            alpha, 
            norm_module=self.norm_module,)
        
        self.layer1 = ResStage(
            dim_in=[width_per_group + width_per_group // out_dim_ratio, 
                    width_per_group // beta_inv,],
            dim_out=[width_per_group * r2, width_per_group * r2 // beta_inv],
            dim_inner=[dim_inner, dim_inner // beta_inv],
            temp_kernel_sizes=temp_kernel[1],
            stride=[1, 1],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=[3,3],
            nonlocal_inds=[[], []],
            nonlocal_group=[[1,1]],
            nonlocal_pool=[],
            instantiation="softmax",
            trans_func_name="basic_transform",
            dilation=[1,1],
            norm_module=self.norm_module,)
        self.layer1_fuse = FuseFastToSlow(
            width_per_group * r2 // beta_inv,
            fusion_conv_channel_ratio,
            fusion_kernel_size,
            alpha,
            norm_module=self.norm_module,)
        
        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)
            
        self.layer2 = ResStage(
            dim_in=[width_per_group * r2 + width_per_group * r2 // out_dim_ratio, 
                    width_per_group * r2 // beta_inv,],
            dim_out=[width_per_group * r3, width_per_group * r3 // beta_inv],
            dim_inner=[dim_inner * 2, dim_inner * 2 // beta_inv],
            temp_kernel_sizes=temp_kernel[2],
            stride=[2, 2],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=[4,4],
            nonlocal_inds=[[], []],
            nonlocal_group=[[1,1]],
            nonlocal_pool=[],
            instantiation="softmax",
            trans_func_name="basic_transform",
            dilation=[1,1],
            norm_module=self.norm_module,
        )
        self.layer2_fuse = FuseFastToSlow(
            width_per_group * r3 // beta_inv,
            fusion_conv_channel_ratio,
            fusion_kernel_size,
            alpha,
            norm_module=self.norm_module,
        )
        
        self.layer3 = ResStage(
            dim_in=[width_per_group * r3 + width_per_group * r3 // out_dim_ratio, 
                    width_per_group * r3 // beta_inv,],
            dim_out=[width_per_group * r4, width_per_group * r4 // beta_inv],
            dim_inner=[dim_inner * 4, dim_inner * 4 // beta_inv],
            temp_kernel_sizes=temp_kernel[3],
            stride=[2, 2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=[6,6],
            nonlocal_inds=[[], []],
            nonlocal_group=[[1,1]],
            nonlocal_pool=[],
            instantiation="softmax",
            trans_func_name="basic_transform",
            dilation=[1,1],
            norm_module=self.norm_module,
        )
        self.layer3_fuse = FuseFastToSlow(
            width_per_group * r4 // beta_inv,
            fusion_conv_channel_ratio,
            fusion_kernel_size,
            alpha,
            norm_module=self.norm_module,
        )
        
        self.layer4 = ResStage(
            dim_in=[width_per_group * r4 + width_per_group * r4 // out_dim_ratio, 
                    width_per_group * r4 // beta_inv,],
            dim_out=[width_per_group * r5, width_per_group * r5 // beta_inv],
            dim_inner=[dim_inner * 8, dim_inner * 8 // beta_inv],
            temp_kernel_sizes=temp_kernel[4],
            stride=[2, 2],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=[3,3],
            nonlocal_inds=[[], []],
            nonlocal_group=[[1,1]],
            nonlocal_pool=[],
            instantiation="softmax",
            trans_func_name="basic_transform",
            dilation=[1,1],
            norm_module=self.norm_module,
        )
    
    def forward(self, x):
        x = x[:]
        x = self.stem(x)
        x = self.stem_fuse(x)
        x = self.layer1(x)
        x = self.layer1_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.layer2(x)
        x = self.layer2_fuse(x)
        x = self.layer3(x)
        x = self.layer3_fuse(x)
        x = self.layer4(x)
        return x
        

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SlowFast().to(device)
data = [[torch.randn(1, 3, 4, 224, 224).to(device), torch.randn(1, 3, 32, 224, 224).to(device)]]

summary(model, input_data=data, col_names=["kernel_size", "input_size", "output_size", "num_params"],)
