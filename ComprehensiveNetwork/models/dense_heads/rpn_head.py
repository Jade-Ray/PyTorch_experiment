import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_head import AnchorHead
from ComprehensiveNetwork.models.cnn import normal_init


class RPNHead(AnchorHead):
    """RPN head.
    Args:
        in_channels (int): Number of channels in the input feature map.
    """
    
    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(1, in_channels, **kwargs)
    
    def _init_layers(self):
        """Initial layers of the head."""
        self.rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels, 
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
    
    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature map of a single scale level"""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def _get_bboxes(self, cls_scores, bbox_preds, mlvl_anchors, img_shapes, scale_factors,
                    rescale=False):
        """Transform outputs for a single batch item into bbox predictions.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                (height, width, 3).
            scale_factors (list[ndarray]): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.
        """
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        batch_size = cls_scores[0].shape[0]
        nms_pre = 1000
        max_per_img = 1000
        nms_pre_tensor = torch.tensor(nms_pre, device=cls_scores[0].device, dtype=torch.long)
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)
            rpn_cls_score = rpn_cls_score.reshape(batch_size, -1)
            scores = rpn_cls_score.sigmoid() # batch_size, H * W * num_anchors * num_classes
            rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            anchors = mlvl_anchors[idx]
            anchors = anchors.expand_as(rpn_bbox_pred) # batch_size, num_total_anchors, 4
            if nms_pre_tensor > 0:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                # keep topk op for dynamic k in onnx model
                if scores.shape[-1] > nms_pre:
                    ranked_scores, rank_inds = scores.sort(descending=True)
                    topk_inds = rank_inds[:, :nms_pre]
                    scores = ranked_scores[:, :nms_pre]
                    batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds)
                    rpn_bbox_pred = rpn_bbox_pred[batch_inds, topk_inds, :]
                    anchors = anchors[batch_inds, topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(scores.new_full((batch_size, scores.size(1))), idx, dtype=torch.long)
            
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_anchors = torch.cat(mlvl_valid_anchors, dim=1)
        batch_mlvl_rpn_bbox_pred = torch.cat(mlvl_bbox_preds, dim=1)
        batch_mlvl_proposals = self.bbox_coder.decode(
            batch_mlvl_anchors, batch_mlvl_rpn_bbox_pred, max_shape=img_shapes)
        batch_mlvl_ids = torch.cat(level_ids, dim=1)
        
        result_list = []
        for (mlvl_proposals, mlvl_scores, mlvl_ids) in zip(batch_mlvl_proposals, batch_mlvl_scores,
                                                           batch_mlvl_ids):
            dets, keep = batched_nms(mlvl_proposals, mlvl_scores, mlvl_ids,
                                     dict(iou_threshold=0.7))
            result_list.append(dets[:max_per_img])
        return result_list
    