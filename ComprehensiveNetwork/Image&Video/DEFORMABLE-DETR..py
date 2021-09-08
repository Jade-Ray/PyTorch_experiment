# %% [markdown]
# # Deformable DETR
# DETR with deformable attention module to speed up convergence and mulit-resolution feature. refer from offical github repo.

# %%
import math
import copy
from typing import Optional, Dict, List

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter


# %% [markdown]
# ## Generate deformabale-detr Model
# with default hidden dim 256

# %%
HIDDEN_DIM = 256
NHEADS = 8
ENC_LAYERS = 6
DEC_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
NUM_FEATURE_LEVELS = 4
ENC_N_POINTS = 4
DEC_N_POINTS = 4
NUM_QUERIES = 300

# %% [markdown]
# #### The part of model of positionEmbed and backbone

# %%
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]   
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes # [max_c, max_h, max_w]

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]) -> None:
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class Backbone(nn.Module):
    """ResNet50 backbone"""
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(pretrained=True, norm_layer=FrozenBatchNorm2d)
        
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.strides = [8, 16, 32]
        self.num_channels = [512, 1024, 2048]
        
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers) # this function return the dict with defined layers

    def forward(self, tensor_list: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        xs = self.body(tensor_list.tensors) # {'0': [bs, 512, h_0, w_0], '1': [bs, 1024, h_1, w_1], '2': [bs, 2048, h_2, w_2]}
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0] # unsample mask with mulit-level size from image mask
            out[name] = NestedTensor(x, mask)
        return out

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats # embed length
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors # [bs, c, h, w]
        mask = tensor_list.mask # [bs, h, w]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32) # 列累加和
        x_embed = not_mask.cumsum(2, dtype=torch.float32) # 行累加和
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) # 特征维度上的索引
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t # [bs, h, w, num_pos_feat]
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) # [bs, h, w, num_pos_feats//2, 2] -> [bs, h, w, num_pos_feat]
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # [bs, h, w, 2 * num_pos_feats] -> [bs, 2 * num_pos_feats, h, w]
        return pos

class Joniner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x) # [nl, bs, num_channel, h, w]
        
        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype)) # [nl, bs, 2 * num_pos_feats, h, w]
        
        return out, pos

# %% [markdown]
# #### The part of model of multi-scale deformable attention
# 对每个query的多头的特征采样点偏置和权重在对尺度和多采样点上进行加权求和

# %%
class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Multi-Scale Deformable Attention Module

        Args:
            d_model (int, optional): hidden dimension. Defaults to 256.
            n_levels (int, optional): number of feature levels. Defaults to 4.
            n_heads (int, optional): number of attention heads. Defaults to 8.
            n_points (int, optional): number of sampling points per attention head per feature level. Defaults to 4.
        """
        
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        
        # 对key值编码到value
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        # 每个query产生对应不同head不同level的偏置
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # 每个偏置的权重
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # 对属于同一个query的来自于不同level的偏置权重在每个head分别归一化
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1) # [(h, w),...] => [(w, h),...]
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            # sampling_locations取值非0~1 此处除以level的宽高到相对坐标
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            # 偏置是相对于目标框的归一化
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4')
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output
        
def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape # batch_size, num_key, num_head, dim
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape # num_query, num_head, num_level, num_point
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1) # 分隔每个level的key
    sampling_grids = 2 * sampling_locations - 1 # pytorch 的 grid_sample 双线性插值需要将采样点映射到-1~1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
            
# %% [markdown]
# #### The part of model of deformable transformer

# %%
class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1, 
                 return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, 
                                                          dropout, num_feature_levels, 
                                                          nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, 
                                                          dropout, num_feature_levels, 
                                                          nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        
        self.reference_points = nn.Linear(d_model, 2)
    
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape # (bs, h, w)
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
        
    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert query_embed is not None
        
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2) # (bs, h*w, c)
            mask = mask.flatten(1) # (bs, h*w)
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # (bs, h*w, c)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1) # (bs, h*w, c) + (1, 1, c), 每一个level提供一个可学习的emb
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device) # (n_level, 2)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # prod 乘积, cumsum 累加
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1) # (bs, n_level, 2)
        
        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        
        # prepare input for decoder
        bs, _, c = memory.shape # bs, n_level*h*w, c
        query_embed, tgt = torch.split(query_embed, c, dim=1) # (Lq, 2*c) => (Lq, c), (Lq, c)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1) # (bs, Lq, c), 每个sample的query相同 参考位置也相同
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid() # 每个query的参考位置是可学习的不同值 (bs, Lq, 2)
        init_reference_out = reference_points
        
        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory, spatial_shapes,
                                            level_start_index, valid_ratios, query_embed, mask_flatten)
        
        inter_references_out = inter_references
        return hs, init_reference_out, inter_references_out, None, None
        
class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # ffn two linear from hid to ffn then to hid
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src
    
    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # ffn
        src = self.forward_ffn(src)
        
        return src

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lv1, (H_, W_) in enumerate(spatial_shapes):
            
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lv1, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lv1, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    
    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            
        return output

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, 
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, 
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # ffn
        tgt = self.forward_ffn(tgt)
        
        return tgt

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, retuen_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.retuen_intermediate = retuen_intermediate

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios, 
                query_pos=None, src_padding_mask=None):
        output = tgt
        
        intermediate = []
        intermediate_reference_points = []
        for _, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None] # (bs, Lq, 1, 2) * (bs, 1, n_level, 2) => (bs, Lq, n_level, 2)
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            
            if self.retuen_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        if self.retuen_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        
        return output, reference_points

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# %% [markdown]
# #### The main of Deformable Detr model

# %%
class DeformableDETR(nn.Module):
    def __init__(self, num_classes, num_queries, num_feature_levels):
        """Initializes the model.

        Args:
            num_classes (int): number of object classes
            num_queries (int): number of object queries
            num_feature_levels (int): num feature lecels
        """
        super().__init__()

        # create ResNet50 backbone
        position_embedding = PositionEmbeddingSine(HIDDEN_DIM // 2, normalize=True)
        backbone = Joniner(Backbone(), position_embedding)
        
        # create deformable transformer
        transformer = DeformableTransformer(HIDDEN_DIM, NHEADS, ENC_LAYERS, DEC_LAYERS, 
                                            DIM_FEEDFORWARD, DROPOUT, False, 
                                            NUM_FEATURE_LEVELS, DEC_N_POINTS, ENC_N_POINTS)
        
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        
        num_backbone_outs = len(backbone.strides)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = backbone.num_channels[_]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim), 
            ))
        for _ in range(num_feature_levels - num_backbone_outs):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim), 
            ))
            in_channels = hidden_dim
        self.input_proj = nn.ModuleList(input_proj_list)
        self.backbone = backbone
        
    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """    
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        
        srcs = []
        masks = []
        
        
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) """     
       
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x