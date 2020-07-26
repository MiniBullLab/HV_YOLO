#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from easyai.loss.utility.base_loss import *
from easyai.loss.utility.anchor_generator import SSDPriorBoxGenerator
from easyai.loss.det2d.utility.det2d_gt_process import Det2dGroundTruthProcess
from easyai.torch_utility.box_utility import torch_corners_box2d_ious, torch_box2d_rect_corner


class MultiBoxLoss(BaseLoss):

    def __init__(self, class_number, iou_threshold, input_size,
                 anchor_counts, anchor_sizes, aspect_ratio_list):
        super().__init__(LossType.MultiBoxLoss)
        self.class_number = class_number
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.anchor_counts = anchor_counts
        self.anchor_sizes = anchor_sizes
        self.aspect_ratio_list = aspect_ratio_list
        self.variances = (0.1, 0.2)

        self.ssd_priorbox = SSDPriorBoxGenerator()
        self.gt_process = Det2dGroundTruthProcess()

        self.info = {'loc_loss': 0.0, 'cls_loss': 0.0}

    def encode(self, gt_boxes, gt_labels, prior_boxes):
        cxcy = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2 - prior_boxes[:, :2]  # [8732,2]
        cxcy /= self.variances[0] * prior_boxes[:, 2:]
        wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]) / prior_boxes[:, 2:]  # [8732,2]
        wh = torch.log(wh) / self.variances[1]
        loc = torch.cat([cxcy, wh], 1)  # [8732, 4]
        cls_conf = 1 + gt_labels  # [8732,], background class = 0
        return loc, cls_conf

    def decode(self, pred_loc, pred_cls, prior_boxes):
        wh = torch.exp(pred_loc[:, :, 2:] * self.variances[1]) * prior_boxes[:, :, 2:]
        cxcy = pred_loc[:, :, :2] * self.variances[0] * prior_boxes[:, :, 2:] + prior_boxes[:, :, :2]
        dets_loc = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 2)  # [b, 8732,4]
        # clip bounding box
        dets_loc[:, :, 0::2] = dets_loc[:, :, 0::2].\
            clamp(min=0, max=self.input_size[0] - 1).div(self.input_size[0])
        dets_loc[:, :, 1::2] = dets_loc[:, :, 1::2].\
            clamp(min=0, max=self.input_size[1] - 1).div(self.input_size[1])
        dets_cls = F.softmax(pred_cls, dim=-1)
        return dets_loc, dets_cls

    def cross_entropy_loss(self, x, y):
        """Cross entropy loss w/o averaging across all samples.

        Args:
          x(tensor): sized [N,D]
          y(tensor): sized [N,]

        Returns:
          (tensor): cross entropy loss, sized [N,]

        """
        xmax = x.data.max()
        log_sum_exp = torch.log(torch.sum(torch.exp(x - xmax), dim=1)) + xmax
        return log_sum_exp.view(-1, 1) - x.gather(1, y.view(-1, 1))

    def hard_negative_mining(self, conf_loss, pos):
        """Return negative indices that is 3x the number as positive indices.

        Args:
          conf_loss: (tensor) cross entropy loss between conf_preds and conf_targets, sized [N*8732,]
          pos: (tensor) positive(matched) box indices, sized [N, 8732]
        Returns:
          (tensor): negative indices, sized [N, 8732]

        """
        batch_size, num_boxes = pos.size()

        conf_loss = conf_loss.view(batch_size, -1)  # [N,8732]
        conf_loss[pos] = 0  # set pos boxes = 0, the rest are neg conf_loss

        _, idx = conf_loss.sort(1, descending=True)  # sort by neg conf_loss
        _, rank = idx.sort(1)  # [N,8732]

        num_pos = pos.long().sum(1)  # [N,1]
        num_neg = torch.clamp(3 * num_pos, min=1, max=num_boxes-1)  # [N,1]
        neg = rank < num_neg.unsqueeze(1).expand_as(rank)  # [N,8732]
        return neg

    def build_targets(self, gt_targets, prior_boxes):
        target_boxes = []
        target_labels = []
        for index in range(len(gt_targets)):
            gt_data = gt_targets[index]
            gt_boxes = self.gt_process.scale_gt_box(gt_data, self.input_size[0], self.input_size[1])
            gt_labels = gt_data[:, 0]
            if gt_data is None or len(gt_data) == 0:
                loc = torch.zeros_like(prior_boxes)
                cls_conf = torch.zeros((prior_boxes.size(0),)).long()
            else:
                tem_boxes = torch_box2d_rect_corner(prior_boxes)
                iou = torch_corners_box2d_ious(gt_boxes, tem_boxes)
                prior_box_iou, max_idx = iou.max(0, keepdim=False)  # [1,8732]
                boxes = gt_boxes[max_idx]  # [8732,4]
                labels = gt_labels[max_idx]  # [8732,],
                loc, cls_conf = self.encode(boxes, labels)
                cls_conf[prior_box_iou < self.iou_threshold] = 0  # background
                # According to IOU, it give every prior box a class label.
                # Then if the IOU is lower than the threshold, the class label is 0(background).
                class_iou, prior_box_idx = iou.max(1, keepdim=False)
                conf_class_idx = prior_box_idx.cpu().numpy()
                cls_conf[conf_class_idx] = gt_labels + 1

            target_boxes.append(loc)
            target_labels.append(cls_conf)

        return torch.stack(target_boxes, 0), torch.stack(target_labels, 0)

    def forward(self, prediction_list, targets=None):
        y_locs = []
        y_confs = []
        anchor_boxes_list = []
        for index, feature_index in enumerate(range(0, len(prediction_list), 2)):
            y_loc = prediction_list[feature_index]
            N, C, H, W = y_loc.size()
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)

            y_conf = prediction_list[feature_index+1]
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, self.class_number)
            y_confs.append(y_conf)

            self.ssd_priorbox.set_anchor_param(self.anchor_counts[index],
                                               (self.anchor_sizes[index], self.anchor_sizes[index+1]),
                                               self.aspect_ratio_list[index])
            anchors = self.ssd_priorbox((W, H), self.input_size)
            anchor_boxes_list.append(anchors)
        # loc_preds(tensor): predicted locations, sized [batch_size, 8732, 4]
        # cls_preds(tensor): predicted class confidences, sized [batch_size, 8732, num_classes]
        loc_preds = torch.cat(y_locs, 1)
        cls_preds = torch.cat(y_confs, 1)
        anchor_boxes = torch.cat(anchor_boxes_list, 0)
        if targets is None:
            prior_boxes = anchor_boxes.unsqueeze(0).repeat(loc_preds.size(0), 1, 1).to(loc_preds.device)
            dets_loc, dets_cls = self.decode(loc_preds, cls_preds, prior_boxes)
            pred_result = torch.cat([dets_loc, dets_cls], 2)
            return pred_result
        else:
            prior_boxes = anchor_boxes.to(loc_preds.device)
            # loc_targets(tensor): encoded target locations, sized [batch_size, 8732, 4]
            # cls_targets:(tensor): encoded target classes, sized [batch_size, 8732]
            loc_targets, cls_targets = self.build_targets(targets, prior_boxes)
            batch_size, num_boxes, _ = loc_preds.size()
            pos = cls_targets > 0  # [N, 8732], pos means the box matched.
            num_matched_boxes = pos.data.float().sum()
            if num_matched_boxes == 0:
                print("No matched boxes")
            # loc_loss
            pos_mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N, 8732, 4]
            pos_loc_preds = loc_preds[pos_mask].view(-1, 4)  # [pos,4]
            pos_loc_targets = loc_targets[pos_mask].view(-1, 4)  # [pos,4]
            loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, reduction='sum')

            # cls_loss
            cls_loss = self.cross_entropy_loss(cls_preds.view(-1, self.num_classes),
                                               cls_targets.view(-1))  # [N*8732,]
            neg = self.hard_negative_mining(cls_loss, pos)  # [N,8732]
            pos_mask = pos.unsqueeze(2).expand_as(cls_preds)  # [N,8732,21]
            neg_mask = neg.unsqueeze(2).expand_as(cls_preds)  # [N,8732,21]
            mask = (pos_mask + neg_mask).gt(0)
            pos_and_neg = (pos + neg).gt(0)
            preds = cls_preds[mask].view(-1, self.num_classes)  # [pos + neg,21]
            targets = cls_targets[pos_and_neg]  # [pos + neg,]
            cls_loss = F.cross_entropy(preds, targets, reduction='sum', ignore_index=-1)

            if num_matched_boxes > 0:
                loc_loss = loc_loss / num_matched_boxes
                cls_loss = cls_loss / num_matched_boxes

            self.info['loc_loss'] = float(loc_loss.item())
            self.info['cls_loss'] = float(cls_loss.item())

            return loc_loss + cls_loss
