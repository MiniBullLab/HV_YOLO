#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.utility.yolo_loss import YoloLoss
from easyai.loss.cls.label_smoothing_ce2d_loss import LabelSmoothCE2dLoss
from easyai.torch_utility.box_utility import torch_rect_box_ious
import math


__all__ = ['YoloV3Loss']


class YoloV3Loss(YoloLoss):

    def __init__(self, class_number, anchor_sizes, reduction,
                 coord_weight=1.0, noobject_weight=1.0,
                 object_weight=1.0, class_weight=1.0, iou_threshold=0.5):
        super().__init__(LossType.YoloV3Loss, class_number, anchor_sizes)
        self.reduction = reduction
        self.coord_xy_weight = 2.0 * 1.0 * coord_weight
        self.coord_wh_weight = 2.0 * 1.5 * coord_weight
        self.noobject_weight = 1.0 * noobject_weight
        self.object_weight = 1.0 * object_weight
        self.class_weight = 1.0 * class_weight
        self.iou_threshold = iou_threshold

        self.smoothLabel = False
        self.focalLoss = False

        self.anchor_sizes = anchor_sizes / float(self.reduction)

        # criterion
        self.mse_loss = nn.MSELoss(reduce=False)
        self.bce_loss = nn.BCELoss(reduce=False)
        self.smooth_l1_loss = nn.SmoothL1Loss(reduce=False)
        if self.smoothLabel:
            self.ce_loss = LabelSmoothCE2dLoss(class_number, reduction='sum')
        else:
            self.ce_loss = nn.CrossEntropyLoss(size_average=False)

        self.info = {'object_count': 0, 'average_iou': 0, 'recall50': 0, 'recall75': 0,
                     'class': 0.0, 'obj': 0.0, 'no_obj': 0.0,
                     'coord_xy': 0.0, 'coord_wh': 0.0}

    def init_loss(self, device):
        # criteria
        self.mse_loss = self.mse_loss.to(device)
        self.bce_loss = self.bce_loss.to(device)
        self.smooth_l1_loss = self.smooth_l1_loss.to(device)
        self.ce_loss = self.ce.to(device)

    def build_targets(self, pred_boxes, gt_targets, height, width, device):
        """ Compare prediction boxes and ground truths, convert ground truths to network output tensors """
        # Parameters
        batch_size = len(gt_targets)
        nPixels = height * width

        # Tensors
        object_mask = torch.zeros(batch_size, self.anchor_count, nPixels,
                                  requires_grad=False, device=device)
        no_object_mask = torch.ones(batch_size, self.anchor_count, nPixels,
                                    requires_grad=False, device=device)
        coord_mask = torch.zeros(batch_size, self.anchor_count, nPixels, 1,
                                 requires_grad=False, device=device)
        cls_mask = torch.zeros(batch_size, self.anchor_count, nPixels,
                               requires_grad=False, dtype=torch.uint8, device=device)
        tcoord = torch.zeros(batch_size, self.anchor_count, nPixels, 4,
                             requires_grad=False, device=device)
        tconf = torch.zeros(batch_size, self.anchor_count, nPixels,
                            requires_grad=False, device=device)
        tcls = torch.zeros(batch_size, self.anchor_count, nPixels,
                           requires_grad=False, device=device)

        recall50 = 0
        recall75 = 0
        object_count = 0
        iou_sum = 0
        for b in range(batch_size):
            gt_data = gt_targets[b]
            pred_box = pred_boxes[b]
            if len(gt_data) == 0:  # No gt for this image
                continue
            object_count += len(gt_data)
            anchors = self.scale_anchor()
            gt = self.gt_process.scale_gt_box(gt_data, width, height).to(device)

            # Find best anchor for each gt
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = torch_rect_box_ious(gt_wh, anchors)
            _, best_index = iou_gt_anchors.max(1)

            # Set confidence mask of matching detections to 0
            iou_gt_pred = torch_rect_box_ious(gt, pred_box)
            mask = (iou_gt_pred > self.iou_threshold).sum(0) >= 1
            no_object_mask[b][mask.view_as(no_object_mask[b])] = 0

            # Set masks and target values for each gt
            # time consuming
            for i, anno in enumerate(gt_data):
                gi = min(width - 1, max(0, int(gt[i, 0])))
                gj = min(height - 1, max(0, int(gt[i, 1])))
                best_n = best_index[i]
                iou = iou_gt_pred[i][best_n * nPixels + gj * width + gi]
                # debug information
                recall50 += (iou > 0.5).item()
                recall75 += (iou > 0.75).item()
                iou_sum += iou.item()

                object_mask[b][best_n][gj * width + gi] = 1
                no_object_mask[b][best_n][gj * width + gi] = 0
                coord_mask[b][best_n][gj * width + gi][0] = 2 - anno[3] * anno[4]
                tcoord[b][best_n][gj * width + gi][0] = gt[i, 0] - gi
                tcoord[b][best_n][gj * width + gi][1] = gt[i, 1] - gj
                tcoord[b][best_n][gj * width + gi][2] = math.log(gt[i, 2] / self.anchors[best_n, 0])
                tcoord[b][best_n][gj * width + gi][3] = math.log(gt[i, 3] / self.anchors[best_n, 1])
                tconf[b][best_n][gj * width + gi] = 1
                cls_mask[b][best_n][gj * width + gi] = 1
                tcls[b][best_n][gj * width + gi] = anno[0]
        # informaion
        if object_count > 0:
            self.info['object_count'] = object_count
            self.info['average_iou'] = iou_sum / object_count
            self.info['recall50'] = recall50 / object_count
            self.info['recall75'] = recall75 / object_count

        return coord_mask, object_mask, no_object_mask, \
               cls_mask, tcoord, tconf, tcls

    def forward(self, outputs, targets=None):
        """ Compute Yolo loss.
        """
        # Parameters
        batch_size, C, height, width = outputs.size()
        device = outputs.device
        self.anchor_sizes = self.anchor_sizes.to(device)

        outputs = outputs.view(batch_size, self.anchor_count,
                               5 + self.class_number,
                               height, width)
        outputs = outputs.view(batch_size, self.anchor_count, -1,
                               height * width)

        # Get x,y,w,h,conf,cls
        coord = torch.zeros_like(outputs[:, :, :4, :])
        coord[:, :, :2, :] = outputs[:, :, :2, :].sigmoid()  # tx,ty
        coord[:, :, 2:4, :] = outputs[:, :, 2:4, :]  # tw,th
        conf = outputs[:, :, 4, :].sigmoid()
        conf = conf.transpose(2, 3).contiguous().view(batch_size, -1, 1)
        cls = outputs[:, :, 5:, :].transpose(2, 3).contiguous().view(batch_size, -1, self.class_number)
        # Create prediction boxes
        pred_boxes = self.decode_predict_box(coord, batch_size, height, width, device)
        pred_boxes = pred_boxes.view(batch_size, -1, 4)

        if targets is None:
            pred_boxes *= self.reduction
            cls = F.softmax(cls, 2)
            return torch.cat([pred_boxes, conf, cls], 2)
        else:
            coord_mask, object_mask, no_object_mask, \
            cls_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, targets, height, width, device)

            # coord
            # 0 = 1 = 2 = 3, only need first two element
            coord = coord.transpose(2, 3).contiguous()
            coord_mask = coord_mask.expand_as(tcoord)[:, :, :, :2]
            coord_center, tcoord_center = coord[:, :, :, :2], tcoord[:, :, :2]
            coord_wh, tcoord_wh = coord[:, :, :, 2:], tcoord[:, :, :, 2:]
            conf = conf.view(batch_size, self.anchor_count, height * width)
            if self.class_number > 1:
                cls = cls.view(-1, self.class_number)
                tcls = tcls[cls_mask].view(-1).long()
                cls_mask = cls_mask.view(-1, 1).repeat(1, self.class_number)
                cls = cls[cls_mask].view(-1, self.class_number)

            self.init_loss(device)

            # Compute losses
            # x,y BCELoss; w,h SmoothL1Loss, conf BCELoss, class CELoss
            loss_coord_center = self.coord_xy_weight * \
                                (coord_mask * self.bce_loss(coord_center, tcoord_center)).sum()
            loss_coord_wh = self.coord_wh_weight * \
                            (coord_mask * self.smooth_l1_loss(coord_wh, tcoord_wh)).sum()
            loss_coord = loss_coord_center + loss_coord_wh

            loss_conf_pos = self.object_weight * (object_mask * self.bce_loss(conf, tconf)).sum()
            loss_conf_neg = self.noobject_weight * (no_object_mask * self.bce_loss(conf, tconf)).sum()
            loss_conf = loss_conf_pos + loss_conf_neg

            if self.class_number > 1:
                loss_cls = self.class_weight * self.ce_loss(cls, tcls)
                cls_softmax = F.softmax(cls, 1)
                t_ind = torch.unsqueeze(tcls, 1).expand_as(cls_softmax)
                class_prob = torch.gather(cls_softmax, 1, t_ind)[:, 0]
            else:
                loss_cls = torch.tensor(0.0, device=device)
                class_prob = torch.tensor(0.0, device=device)

            if self.info['object_count'] > 0:
                self.info['class'] = class_prob.sum().item() / self.info['object_count']
                self.info['obj'] = (object_mask * conf).sum().item() / self.info['object_count']
                self.info['no_obj'] = (no_object_mask * conf).sum().item() / \
                                      batch_size * self.anchor_count * height * width
                self.info['coord_xy'] = (coord_mask * self.mse_loss(coord_center, tcoord_center)).sum().item() / self.info['object_count']
                self.info['coord_wh'] = (coord_mask * self.mse_loss(coord_wh, tcoord_wh)).sum().item() / self.info['object_count']
            self.printInfo()

            all_loss = loss_coord + loss_conf + loss_cls
            return all_loss
