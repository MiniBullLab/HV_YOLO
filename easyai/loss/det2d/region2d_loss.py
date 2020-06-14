#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.yolo_loss import YoloLoss
from easyai.torch_utility.box_utility import torch_rect_box_ious
import math

__all__ = ['Region2dLoss']


class Region2dLoss(YoloLoss):

    def __init__(self, class_number, anchor_sizes, reduction,
                 coord_weight=1.0, noobject_weight=1.0,
                 object_weight=5.0, class_weight=2.0, iou_threshold=0.6):
        super().__init__(LossType.Region2dLoss, class_number, anchor_sizes)
        self.reduction = reduction
        self.coord_weight = coord_weight
        self.noobject_weight = noobject_weight
        self.object_scale = object_weight
        self.class_weight = class_weight
        self.iou_threshold = iou_threshold

        self.anchor_sizes = anchor_sizes / float(self.reduction)

        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

        self.info = {'object_count': 0, 'average_iou': 0, 'recall50': 0, 'recall75': 0,
                     'coord_xy_loss': 0.0, 'coord_wh_loss': 0.0, 'conf_loss': 0.0,
                     'cls_loss': 0.0}

    def build_targets(self, pred_boxes, gt_targets, height, width, device):
        batch_size = len(gt_targets)

        object_mask = torch.zeros(batch_size, self.anchor_count, height * width,
                                  requires_grad=False, device=device)
        no_object_mask = torch.ones(batch_size, self.anchor_count, height*width,
                                    requires_grad=False, device=device)
        conf_mask = torch.ones(batch_size, self.anchor_count, height * width,
                               requires_grad=False, device=device) * self.noobject_weight
        coord_mask = torch.zeros(batch_size, self.anchor_count, 1, height * width,
                                 requires_grad=False, device=device)
        cls_mask = torch.zeros(batch_size, self.anchor_count, height * width,
                               requires_grad=False, device=device).byte()
        tcoord = torch.zeros(batch_size, self.anchor_count, 4, height * width,
                             requires_grad=False, device=device)
        tconf = torch.zeros(batch_size, self.anchor_count, height * width,
                            requires_grad=False, device=device)
        tcls = torch.zeros(batch_size, self.anchor_count, height * width,
                           requires_grad=False, device=device)

        recall50 = 0
        recall75 = 0
        object_count = 0
        iou_sum = 0
        for b in range(batch_size):
            gt_data = gt_targets[b]
            pre_box = pred_boxes[b]
            if len(gt_data) == 0:
                continue
            object_count += len(gt_data)
            gt_box = self.gt_process.scale_gt_box(gt_data, width, height).to(device)
            anchors = self.scale_anchor()

            # Find best anchor for each ground truth
            gt_wh = gt_box.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = torch_rect_box_ious(gt_wh, anchors)
            _, best_indexs = iou_gt_anchors.max(1)

            # Set confidence mask of matching detections to 0
            ious = torch_rect_box_ious(gt_box, pre_box)
            mask = (ious > self.iou_threshold).sum(0) >= 1
            conf_mask[b][mask.view_as(conf_mask[b])] = 0
            no_object_mask[b][mask.view_as(no_object_mask[b])] = 0

            # Set masks and target values for each ground truth
            for i, anno in enumerate(gt_data):
                gi = min(width - 1, max(0, int(gt_box[i, 0])))
                gj = min(height - 1, max(0, int(gt_box[i, 1])))
                best_n = best_indexs[i]
                iou = ious[i][best_n * height * width + gj * width + gi]

                recall50 += (iou > 0.5).item()
                recall75 += (iou > 0.75).item()
                iou_sum += iou.item()

                coord_mask[b][best_n][0][gj * width + gi] = 1
                cls_mask[b][best_n][gj * width + gi] = 1
                conf_mask[b][best_n][gj * width + gi] = self.object_weight
                object_mask[b][best_n][gj * width + gi] = 1
                no_object_mask[b][best_n][gj * width + gi] = 0
                tcoord[b][best_n][0][gj * width + gi] = gt_box[i, 0] - gi
                tcoord[b][best_n][1][gj * width + gi] = gt_box[i, 1] - gj
                tcoord[b][best_n][2][gj * width + gi] = math.log(max(gt_box[i, 2], 1.0) /
                                                                 self.anchor_sizes[best_n, 0])
                tcoord[b][best_n][3][gj * width + gi] = math.log(max(gt_box[i, 3], 1.0) /
                                                                 self.anchor_sizes[best_n, 1])
                tconf[b][best_n][gj * width + gi] = iou
                tcls[b][best_n][gj * width + gi] = int(anno[0])

        if object_count > 0:
            self.info['object_count'] = object_count
            self.info['average_iou'] = iou_sum / object_count
            self.info['recall50'] = recall50 / object_count
            self.info['recall75'] = recall75 / object_count

        return coord_mask, conf_mask, object_mask, no_object_mask, \
               cls_mask, tcoord, tconf, tcls

    def forward(self, outputs, targets=None):
        batch_size = outputs.data.size(0)
        height = outputs.data.size(2)
        width = outputs.data.size(3)
        device = outputs.device
        self.anchor_sizes = self.anchor_sizes.to(device)

        outputs = outputs.view(batch_size, self.anchor_count,
                               5 + self.class_number,
                               height, width)
        outputs = outputs.view(batch_size, self.anchor_count, -1,
                               height * width).permute(0, 1, 3, 2).contiguous()

        # Get x,y,w,h,conf,cls
        coord = outputs[:, :, :, :4]
        coord[:, :, :, :2] = torch.sigmoid(outputs[:, :, :, :2])
        coord = coord.view(batch_size, -1, 4)
        conf = torch.sigmoid(outputs[:, :, :, 4]).view(batch_size, -1, 1)
        cls = outputs[:, :, :, 5:].view(batch_size, -1, self.class_number)

        # Create prediction boxes
        pred_boxes = self.decode_predict_box(coord, batch_size, height, width, device)
        pred_boxes = pred_boxes.view(batch_size, -1, 4)

        if targets is None:
            cls = F.softmax(cls, 2)
            return torch.cat([pred_boxes, conf, cls], 2)
        else:
            coord_mask, conf_mask, object_mask, no_object_mask, cls_mask, tcoord, tconf, tcls = \
                self.build_targets(pred_boxes, targets, height, width, device)
            coord_mask = coord_mask.expand_as(tcoord)
            tcls = tcls[cls_mask].view(-1).long()
            cls_mask = cls_mask.view(-1, 1).repeat(1, self.class_number)
            cls = cls[cls_mask].view(-1, self.class_number)

            conf_mask = conf_mask.sqrt()

            # Compute loss
            loss_coord = self.coord_weight * self.mse(coord * coord_mask, tcoord * coord_mask)
            loss_conf = self.mse(conf * conf_mask, tconf * conf_mask)
            loss_cls = self.class_weight * self.ce(cls, tcls)
            loss = loss_coord + loss_conf + loss_cls

            if self.info['object_count'] > 0:
                self.info['coord_xy_loss'] = (self.mse(coord[:, :, 0:2, :] *
                                                       coord_mask[:, :, 0:2, :],
                                                       tcoord[:, :, 0:2, :] *
                                                       coord_mask[:, :, 0:2, :])
                                              ).item()
                self.info['coord_wh_loss'] = (self.mse(coord[:, :, 2:4, :] *
                                                       coord_mask[:, :, 2:4, :],
                                                       tcoord[:, :, 2:4, :] *
                                                       coord_mask[:, :, 2:4, :])
                                              ).item()
                self.info['conf_loss'] = loss_conf.item()
                self.info['cls_loss'] = loss_cls.item()
            self.print_info()

            return loss
