#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from loss.base_loss import *
from loss.loss_name import LossType
import math
from loss.smooth_cross_entropy import SmoothCrossEntropy
from loss.focal_loss import FocalBinaryLoss

__all__ = ['YoloLoss']


def bbox_ious(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

    Args:
        boxes1 (torch.Tensor): List of bounding boxes
        boxes2 (torch.Tensor): List of bounding boxes

    Note:
        List format: [[xc, yc, w, h],...]
    """
    b1_len = boxes1.size(0)
    b2_len = boxes2.size(0)

    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions


class YoloLoss(BaseLoss):
    """ Computes yolo loss from darknet network output and target annotation.

    Args:
        num_classes (int): number of categories
        anchors (list): 2D list representing anchor boxes (see :class:`lightnet.network.Darknet`)
        coord_scale (float): weight of bounding box coordinates
        noobject_scale (float): weight of regions without target boxes
        object_scale (float): weight of regions with target boxes
        class_scale (float): weight of categorical predictions
        thresh (float): minimum iou between a predicted box and ground truth for them to be considered matching
        seen (int): How many images the network has already been trained on.
    """
    def __init__(self, num_classes, anchors, anchors_mask, coord_scale=1.0, noobject_scale=1.0,
                 object_scale=1.0, class_scale=1.0, thresh=0.5, smoothLabel=False, focalLoss=False):
        super().__init__(LossType.YoloLoss)

        self.num_classes = num_classes
        self.num_anchors = len(anchors_mask)
        self.anchor_step = len(anchors[0])
        self.anchors_mask = anchors_mask

        self.smoothLabel = smoothLabel
        self.focalLoss = focalLoss

        if anchors_mask[0] == (self.num_anchors * 2):  # 6
            self.reduction = 32
        elif anchors_mask[0] == self.num_anchors:  # 3
            self.reduction = 16
        else:
            self.reduction = 8

        self.anchors = torch.Tensor(anchors) / float(self.reduction)
        self.seen = 0

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale

        self.thresh = thresh

        self.info = {'avg_iou': 0, 'class': 0, 'obj': 0, 'no_obj': 0, 
                'recall50': 0, 'recall75': 0, 'obj_cur': 0, 'obj_all': 0,
                'coord_xy': 0, 'coord_wh': 0}

        # criterion
        self.mse = nn.MSELoss(reduce=False)
        self.bce = nn.BCELoss(reduce=False)
        self.smooth_l1 = nn.SmoothL1Loss(reduce=False)
        if self.smoothLabel:
            self.ce = SmoothCrossEntropy(reduction='sum')
        else:
            self.ce = nn.CrossEntropyLoss(size_average=False)
        self.fl_bce = FocalBinaryLoss(gamma=2, reduce=False)

    def forward(self, output, target=None, seen=None):
        """ Compute Yolo loss.
        """
        # Parameters
        nB = output.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.size(2)
        nW = output.size(3)
        device = output.device

        if seen is not None:
            self.seen = seen
        else:
            self.seen += nB

        self.anchors = self.anchors.to(device)

        # Get x,y,w,h,conf,cls
        output = output.view(nB, nA, -1, nH*nW)
        coord = torch.zeros_like(output[:, :, :4])
        coord[:, :, :2] = output[:, :, :2].sigmoid()    # tx,ty
        coord[:, :, 2:4] = output[:, :, 2:4]            # tw,th
        conf = output[:, :, 4].sigmoid()
        if nC > 1:
            cls = output[:, :, 5:].contiguous().view(nB*nA, nC, nH*nW).transpose(1, 2).contiguous().view(-1, nC)

        # Create prediction boxes
        # time consuming
        pred_boxes = torch.zeros(nB*nA*nH*nW, 4, dtype=torch.float, device=device)
        lin_x = torch.linspace(0, nW-1, nW).to(device).repeat(nH, 1).view(nH*nW)
        lin_y = torch.linspace(0, nH-1, nH).to(device).repeat(nW, 1).t().contiguous().view(nH*nW)
        anchor_w = self.anchors[self.anchors_mask, 0].view(nA, 1).to(device)
        anchor_h = self.anchors[self.anchors_mask, 1].view(nA, 1).to(device)

        pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)
        pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)
        pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
        pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)

        if target is not None:
            # Get target values
            coord_mask, conf_pos_mask, conf_neg_mask, cls_mask, tcoord, tconf, tcls = \
                self.build_targets(pred_boxes, target, nH, nW)
            # coord
            # 将coord_mask扩充到和tcoord一样的维度，再取前两维
            coord_mask = coord_mask.expand_as(tcoord)[:,:,:2] # 0 = 1 = 2 = 3, only need first two element
            coord_center, tcoord_center = coord[:,:,:2], tcoord[:,:,:2]
            coord_wh, tcoord_wh = coord[:,:,2:], tcoord[:,:,2:]
            if nC > 1:
                # tcls变成一维数组, 而cls还是变成[xx, nC]维
                tcls = tcls[cls_mask].view(-1).long()
                # label smooth
                if self.smoothLabel:
                    smoothL = self.ce.smooth_label(tcls, self.num_classes)

                cls_mask = cls_mask.view(-1, 1).repeat(1, nC)
                cls = cls[cls_mask].view(-1, nC)

            # criteria
            self.bce = self.bce.to(device)
            self.mse = self.mse.to(device)
            self.smooth_l1 = self.smooth_l1.to(device)
            self.ce = self.ce.to(device)
            self.fl_bce = self.fl_bce.to(device)

            bce = self.bce
            mse = self.mse
            smooth_l1 = self.smooth_l1
            ce = self.ce
            fl_bce = self.fl_bce

            # Compute losses
            # x,y BCELoss; w,h SmoothL1Loss, conf BCELoss, class CELoss
            loss_coord_center = 2.0 * 1.0 * self.coord_scale * (coord_mask*bce(coord_center, tcoord_center)).sum()
            loss_coord_wh = 2.0 * 1.5 * self.coord_scale * (coord_mask*smooth_l1(coord_wh, tcoord_wh)).sum()
            self.loss_coord = loss_coord_center + loss_coord_wh

            if self.focalLoss:
                loss_conf_pos = 1.0 * self.object_scale * (conf_pos_mask * fl_bce(conf, tconf)).sum()
                loss_conf_neg = 1.0 * self.noobject_scale * (conf_neg_mask * fl_bce(conf, tconf)).sum()
                self.loss_conf = loss_conf_pos + loss_conf_neg
            else:
                loss_conf_pos = 1.0 * self.object_scale * (conf_pos_mask * bce(conf, tconf)).sum()
                loss_conf_neg = 1.0 * self.noobject_scale * (conf_neg_mask * bce(conf, tconf)).sum()
                self.loss_conf = loss_conf_pos + loss_conf_neg

            if nC > 1 and cls.numel() > 0:
                if self.smoothLabel:
                    self.loss_cls = self.class_scale * 1.0 * ce(cls, smoothL)
                else:
                    self.loss_cls = self.class_scale * 1.0 * ce(cls, tcls)
                cls_softmax = F.softmax(cls, 1)
                t_ind = torch.unsqueeze(tcls, 1).expand_as(cls_softmax)
                class_prob = torch.gather(cls_softmax, 1, t_ind)[:, 0]
            else:
                self.loss_cls = torch.tensor(0.0, device=device)
                class_prob = torch.tensor(0.0, device=device)

            obj_cur = max(self.info['obj_cur'], 1)
            self.info['class'] = class_prob.sum().item() / obj_cur
            self.info['obj'] = (conf_pos_mask*conf).sum().item() / obj_cur
            self.info['no_obj'] = (conf_neg_mask*conf).sum().item() / output.numel()
            self.info['coord_xy'] = (coord_mask * mse(coord_center, tcoord_center)).sum().item() / obj_cur
            self.info['coord_wh'] = (coord_mask* mse(coord_wh, tcoord_wh)).sum().item() / obj_cur
            self.printInfo()

            self.loss_tot = (self.loss_coord + self.loss_conf + self.loss_cls)
            return self.loss_tot

        else:
            pred_boxes = pred_boxes.view(nB, -1, 4)
            pred_boxes*= self.reduction
            conf = conf.view(nB, -1, 1)
            cls = output[:, :, 5:].contiguous().view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(nB, -1, nC)
            cls = F.softmax(cls, 2)

            return torch.cat([pred_boxes, conf, cls], 2)

    def build_targets(self, pred_boxes, ground_truth, nH, nW):
        """ Compare prediction boxes and targets, convert targets to network output tensors """
        return self.__build_targets_brambox(pred_boxes, ground_truth, nH, nW)

    def __build_targets_brambox(self, pred_boxes, ground_truth, nH, nW):
        """ Compare prediction boxes and ground truths, convert ground truths to network output tensors """
        # Parameters
        nB = len(ground_truth)
        nA = self.num_anchors
        nAnchors = nA*nH*nW
        nPixels = nH*nW
        device = pred_boxes.device

        # Tensors
        conf_pos_mask = torch.zeros(nB, nA, nH*nW, requires_grad=False, device=device)
        conf_neg_mask = torch.ones(nB, nA, nH*nW, requires_grad=False, device=device)
        coord_mask = torch.zeros(nB, nA, 1, nH*nW, requires_grad=False, device=device)
        cls_mask = torch.zeros(nB, nA, nH*nW, requires_grad=False, dtype=torch.uint8, device=device)
        tcoord = torch.zeros(nB, nA, 4, nH*nW, requires_grad=False, device=device)
        tconf = torch.zeros(nB, nA, nH*nW, requires_grad=False, device=device)
        tcls = torch.zeros(nB, nA, nH*nW, requires_grad=False, device=device)

        recall50 = 0
        recall75 = 0
        obj_all = 0
        obj_cur = 0
        iou_sum = 0
        for b in range(nB):
            if len(ground_truth[b]) == 0:   # No gt for this image
                continue

            obj_all += len(ground_truth[b])

            # Build up tensors
            cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors]
            if self.anchor_step == 4:
                anchors = self.anchors.clone()
                anchors[:, :2] = 0
            else:
                anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)
            gt = torch.zeros(len(ground_truth[b]), 4, device=device)
            for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = anno[1] * nW
                gt[i, 1] = anno[2] * nH
                gt[i, 2] = anno[3] * nW
                gt[i, 3] = anno[4] * nH

            # Set confidence mask of matching detections to 0
            iou_gt_pred = bbox_ious(gt, cur_pred_boxes)
            mask = (iou_gt_pred > self.thresh).sum(0) >= 1
            conf_neg_mask[b][mask.view_as(conf_neg_mask[b])] = 0
            
            # Find best anchor for each gt
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = bbox_ious(gt_wh, anchors)
            _, best_anchors = iou_gt_anchors.max(1)

            # Set masks and target values for each gt
            # time consuming
            for i, anno in enumerate(ground_truth[b]):
                gi = min(nW-1, max(0, int(gt[i, 0])))
                gj = min(nH-1, max(0, int(gt[i, 1])))
                cur_n = best_anchors[i]
                if cur_n in self.anchors_mask:
                    best_n = self.anchors_mask.index(cur_n)
                else:
                    continue

                iou = iou_gt_pred[i][best_n*nPixels+gj*nW+gi]
                # debug information
                obj_cur += 1
                recall50 += (iou > 0.5).item()
                recall75 += (iou > 0.75).item()
                iou_sum += iou.item()

                coord_mask[b][best_n][0][gj*nW+gi] = 2 - anno[3]*anno[4]/(nW*nH*self.reduction*self.reduction)
                cls_mask[b][best_n][gj*nW+gi] = 1
                conf_pos_mask[b][best_n][gj*nW+gi] = 1 
                conf_neg_mask[b][best_n][gj*nW+gi] = 0
                tcoord[b][best_n][0][gj*nW+gi] = gt[i, 0] - gi
                tcoord[b][best_n][1][gj*nW+gi] = gt[i, 1] - gj
                tcoord[b][best_n][2][gj*nW+gi] = math.log(gt[i, 2]/self.anchors[cur_n, 0])
                tcoord[b][best_n][3][gj*nW+gi] = math.log(gt[i, 3]/self.anchors[cur_n, 1])
                tconf[b][best_n][gj*nW+gi] = 1
                tcls[b][best_n][gj*nW+gi] = anno[0]
        # loss informaion
        self.info['obj_cur'] = obj_cur
        self.info['obj_all'] = obj_all
        if obj_cur == 0: 
            obj_cur = 1
        self.info['avg_iou'] = iou_sum / obj_cur 
        self.info['recall50'] = recall50 / obj_cur
        self.info['recall75'] = recall75 / obj_cur

        return coord_mask, conf_pos_mask, conf_neg_mask, cls_mask, tcoord, tconf, tcls

    def printInfo(self):
        info = self.info
        info_str = 'AVG IOU %.4f, Class %.4f, Obj %.4f, No obj %.4f, ' \
                '.5R %.4f, .75R %.4f, Cur obj %3d, All obj %3d, Coord xy %.4f, Coord wh %.4f' % \
                (info['avg_iou'], info['class'], info['obj'], info['no_obj'],
                        info['recall50'], info['recall75'], info['obj_cur'], info['obj_all'],
                        info['coord_xy'], info['coord_wh'])
        print('%s' % info_str)

        # reset
        self.info = {'avg_iou': 0, 'class': 0, 'obj': 0, 'no_obj': 0, 
                'recall50': 0, 'recall75': 0, 'obj_cur': 0, 'obj_all': 0,
                'coord_xy': 0, 'coord_wh': 0}


