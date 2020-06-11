#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.yolo_loss import YoloLoss
import math
from easyai.loss.cls.smooth_cross_entropy import SmoothCrossEntropy
from easyai.loss.det2d.focal_loss import FocalBinaryLoss
from easyai.torch_utility.box_utility import torch_rect_box_ious


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
            self.ce_loss = SmoothCrossEntropy(reduction='sum')
        else:
            self.ce_loss = nn.CrossEntropyLoss(size_average=False)
        self.fl_bce_loss = FocalBinaryLoss(gamma=2, reduce=False)

        self.info = {'avg_iou': 0, 'class': 0, 'obj': 0, 'no_obj': 0,
                     'recall50': 0, 'recall75': 0, 'obj_cur': 0, 'obj_all': 0,
                     'coord_xy': 0, 'coord_wh': 0}

    def init_loss(self, device):
        # criteria
        self.mse_loss = self.mse_loss.to(device)
        self.bce_loss = self.bce_loss.to(device)
        self.smooth_l1_loss = self.smooth_l1_loss.to(device)
        self.ce_loss = self.ce.to(device)
        self.fl_bce_loss = self.fl_bce.to(device)

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
        coord_mask = torch.zeros(batch_size, self.anchor_count, 1, nPixels,
                                 requires_grad=False, device=device)
        cls_mask = torch.zeros(batch_size, self.anchor_count, nPixels,
                               requires_grad=False, dtype=torch.uint8, device=device)
        tcoord = torch.zeros(batch_size, self.anchor_count, 4, nPixels,
                             requires_grad=False, device=device)
        tconf = torch.zeros(batch_size, self.anchor_count, nPixels,
                            requires_grad=False, device=device)
        tcls = torch.zeros(batch_size, self.anchor_count, nPixels,
                           requires_grad=False, device=device)

        recall50 = 0
        recall75 = 0
        obj_all = 0
        obj_cur = 0
        iou_sum = 0
        for b in range(batch_size):
            gt_data = gt_targets[b]
            pred_box = pred_boxes[b]

            if len(gt_data) == 0:  # No gt for this image
                continue

            obj_all += len(gt_data)

            anchors = self.scale_anchor()
            gt = self.gt_process.scale_gt_box(gt_data, width, height).to(device)

            # Find best anchor for each gt
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = torch_rect_box_ious(gt_wh, anchors)
            _, best_index = iou_gt_anchors.max(1)

            # Set confidence mask of matching detections to 0
            iou_gt_pred = torch_rect_box_ious(gt, pred_box)
            mask = (iou_gt_pred > self.thresh).sum(0) >= 1
            no_object_mask[b][mask.view_as(no_object_mask[b])] = 0

            # Set masks and target values for each gt
            # time consuming
            for i, anno in enumerate(gt_data):
                gi = min(width - 1, max(0, int(gt[i, 0])))
                gj = min(height - 1, max(0, int(gt[i, 1])))
                best_n = best_index[i]
                iou = iou_gt_pred[i][best_n * nPixels + gj * width + gi]
                # debug information
                obj_cur += 1
                recall50 += (iou > 0.5).item()
                recall75 += (iou > 0.75).item()
                iou_sum += iou.item()

                coord_mask[b][best_n][0][gj * width + gi] = 2 - anno[3] * anno[4] / (
                            width * height * self.reduction * self.reduction)
                cls_mask[b][best_n][gj * width + gi] = 1
                object_mask[b][best_n][gj * width + gi] = 1
                no_object_mask[b][best_n][gj * width + gi] = 0
                tcoord[b][best_n][0][gj * width + gi] = gt[i, 0] - gi
                tcoord[b][best_n][1][gj * width + gi] = gt[i, 1] - gj
                tcoord[b][best_n][2][gj * width + gi] = math.log(gt[i, 2] / self.anchors[best_n, 0])
                tcoord[b][best_n][3][gj * width + gi] = math.log(gt[i, 3] / self.anchors[best_n, 1])
                tconf[b][best_n][gj * width + gi] = 1
                tcls[b][best_n][gj * width + gi] = anno[0]
        # loss informaion
        self.info['obj_cur'] = obj_cur
        self.info['obj_all'] = obj_all
        if obj_cur == 0:
            obj_cur = 1
        self.info['avg_iou'] = iou_sum / obj_cur
        self.info['recall50'] = recall50 / obj_cur
        self.info['recall75'] = recall75 / obj_cur

        return coord_mask, object_mask, no_object_mask, \
               cls_mask, tcoord, tconf, tcls

    def forward(self, outputs, targets=None):
        """ Compute Yolo loss.
        """
        # Parameters
        N, C, H, W = outputs.size()
        device = outputs.device
        self.anchor_sizes = self.anchor_sizes.to(device)

        outputs = outputs.view(N, self.anchor_count,
                               5 + self.class_number,
                               H, W)
        outputs = outputs.view(N, self.anchor_count, -1,
                               H * W).permute(0, 1, 3, 2).contiguous()

        # Get x,y,w,h,conf,cls
        coord = outputs[:, :, :, :4]  # tx, ty, tw, th
        coord[:, :, :2, :] = torch.sigmoid(outputs[:, :, :, :2]) # tx,ty
        coord = coord.view(N, -1, 4)
        conf = torch.sigmoid(outputs[:, :, :, 4]).view(N, -1, 1)
        cls = outputs[:, :, :, 5:].view(N, -1, self.class_number)

        # Create prediction boxes
        pred_boxes = self.decode_predict_box(coord, H, H, W, device)
        pred_boxes = pred_boxes.view(N, -1, 4)

        if targets is None:
            pred_boxes *= self.reduction
            cls = F.softmax(cls, 2)
            return torch.cat([pred_boxes, conf, cls], 2)
        else:
            coord_mask, object_mask, no_object_mask, \
            cls_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, targets, H, W, device)

            # coord
            # 0 = 1 = 2 = 3, only need first two element
            coord_mask = coord_mask.expand_as(tcoord)[:, :, :2]
            coord_center, tcoord_center = coord[:, :, :2], tcoord[:, :, :2]
            coord_wh, tcoord_wh = coord[:, :, 2:], tcoord[:, :, 2:]
            if self.class_number > 1:
                # tcls变成一维数组, 而cls还是变成[xx, nC]维
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

            obj_cur = max(self.info['obj_cur'], 1)
            self.info['class'] = class_prob.sum().item() / obj_cur
            self.info['obj'] = (object_mask * conf).sum().item() / obj_cur
            self.info['no_obj'] = (no_object_mask * conf).sum().item() / N * self.anchor_count * H * W
            self.info['coord_xy'] = (coord_mask * self.mse_loss(coord_center, tcoord_center)).sum().item() / obj_cur
            self.info['coord_wh'] = (coord_mask * self.mse_loss(coord_wh, tcoord_wh)).sum().item() / obj_cur
            self.printInfo()

            all_loss = loss_coord + loss_conf + loss_cls
            return all_loss
