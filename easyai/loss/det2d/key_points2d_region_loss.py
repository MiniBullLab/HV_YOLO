#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.utility.yolo_loss import YoloLoss


class KeyPoints2dRegionLoss(YoloLoss):
    def __init__(self, class_number, point_count,
                 coord_weight=1.0/2.0, noobject_weight=1.0,
                 object_weight=5.0, class_weight=2.0, iou_threshold=0.6):
        super().__init__(LossType.KeyPoints2dRegionLoss, class_number)
        self.point_count = point_count
        self.loc_count = point_count * 2
        self.coord_scale = coord_weight
        self.noobject_scale = noobject_weight
        self.object_scale = object_weight
        self.class_scale = class_weight
        self.threshold = iou_threshold

        self.seen = 0
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.info = {'num_ground_truth': 0, 'num_det_correct': 0,
                     'x_loss': 0.0, 'y_loss': 0.0, 'conf_loss': 0.0, 'cls_loss': 0.0}

    def normaliza_points(self, pred_corners, batch_size, H, W):
        for index in range(batch_size):
            for i in range(0, self.point_count, 2):
                pred_corners[index, :, i] = pred_corners[i] / W
                pred_corners[index, :, i + 1] = pred_corners[i + 1] / H

    def corner2d_confidence(self, gt_corners, pr_corners, points_count,
                            width, height, threshold=80, sharpness=2):
        '''
        gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (18,) type: list
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (18,), type: list
        threshold: distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a list of shape (points_count,) with 9 confidence values
        '''

        if len(gt_corners.shape) < 2 or len(pr_corners.shape) < 2:
            gt_corners = torch.unsqueeze(gt_corners, 1)
            pr_corners = torch.unsqueeze(pr_corners, 1)

        shape = gt_corners.size()
        nA = shape[1]
        dist = gt_corners - pr_corners
        dist = dist.t().contiguous().view(nA, points_count, 2)
        dist[:, :, 0] = dist[:, :, 0] * width
        dist[:, :, 1] = dist[:, :, 1] * height

        dist_thresh = torch.FloatTensor([threshold]).repeat(nA, points_count).to(gt_corners.device)
        dist = torch.sqrt(torch.sum((dist) ** 2, dim=2))
        mask = (dist < dist_thresh).type_as(gt_corners)
        # mask * (torch.exp(math.log(2) * (1.0 - dist/rrt)) - 1)
        conf = torch.exp(sharpness * (1 - dist / dist_thresh)) - 1
        conf0 = torch.exp(sharpness * (1 - torch.zeros(conf.size(0), 1))) - 1
        conf = conf / conf0.repeat(1, 9)
        # conf = 1 - dist/distthresh
        conf = mask * conf  # nA x 9
        mean_conf = torch.mean(conf, dim=1)

        return mean_conf

    def compute_confidence(self, pred_corners, gt_targets,
                           batch_size, H, W, device):
        num_anchors = H * W
        conf_mask = torch.ones(batch_size, num_anchors, requires_grad=False,
                               device=device) * self.noobject_scale
        for b in range(batch_size):
            pred_data = pred_corners[b]
            gt_data = gt_targets[b]
            if len(gt_data) == 0:
                continue
            cur_confs = torch.zeros(num_anchors)
            for i, anno in enumerate(gt_data):
                gt_corners, _, _ = self.gt_process.scale_gt_points(anno, self.point_count, W, H).\
                    repeat(num_anchors, 1).t().to(device)
                mean_conf = self.corner2d_confidence(gt_corners, pred_data.t(), self.point_count, W, H)
                # some irrelevant areas are filtered, in the same grid multiple anchor boxes might exceed the threshold
                cur_confs = torch.max(cur_confs, mean_conf).view(1, H, W)
            conf_mask[b][cur_confs > self.threshold] = 0
        return conf_mask

    def seen_process(self, batch_size, tx, ty, coord_mask):
        self.seen += batch_size
        if self.seen < -1:  # 6400:
            for i in range(self.point_count):
                tx[i].fill_(0.5)
                ty[i].fill_(0.5)
            coord_mask.fill_(1)

    def build_targets(self, pred_corners, gt_targets, H, W, device):
        batch_size = len(gt_targets)
        coord_mask = torch.zeros(batch_size, H * W, requires_grad=False, device=device)
        cls_mask = torch.zeros(batch_size, H * W, requires_grad=False, device=device)

        txs = []
        tys = []
        for i in range(self.point_count):
            txs.append(torch.zeros(batch_size, H * W, device=device))
            tys.append(torch.zeros(batch_size, H * W, device=device))
        tconf = torch.zeros(batch_size, H * W, device=device)
        tcls = torch.zeros(batch_size, H * W, device=device)

        self.normaliza_points(pred_corners, batch_size, H, W)
        conf_mask = self.compute_confidence(pred_corners, gt_targets,
                                            batch_size, H, W, device)
        # self.seen_process(batch_size, txs, tys, coord_mask)

        num_ground_truth = 0
        num_det_correct = 0
        for b in range(batch_size):
            pred_data = pred_corners[b]
            gt_data = gt_targets[b]
            if len(gt_data) == 0:
                continue
            num_ground_truth += len(gt_data)
            for i, anno in enumerate(gt_data):
                gt_corners, gx, gy = self.gt_process.scale_gt_points(anno, self.point_count, W, H)
                gi0 = int(gx[0])
                gj0 = int(gy[0])
                gt_corners = gt_corners.to(device)
                pred_corners = pred_data[gj0 * W + gi0]
                conf = self.corner2d_confidence(gt_corners, pred_corners, self.point_count, W, H)
                coord_mask[b][gj0 * W + gi0] = 1
                cls_mask[b][gj0 * W + gi0] = 1
                conf_mask[b][gj0 * W + gi0] = self.object_scale
                # Update targets
                for index in range(self.point_count):
                    txs[index][b][gj0 * W + gi0] = gx[index] - gi0
                    tys[index][b][gj0 * W + gi0] = gy[index] - gj0
                tconf[b][gj0 * W + gi0] = conf
                tcls[b][gj0 * W + gi0] = gt_data[i, 0]
                # Update recall during training
                if conf > 0.5:
                    num_det_correct = num_det_correct + 1
        return num_ground_truth, num_det_correct, \
               coord_mask, conf_mask, cls_mask, \
               txs, tys, tconf, tcls

    def forward(self, outputs, targets=None):
        N, C, H, W = outputs.size()
        device = outputs.device
        output = outputs.view(N, (self.loc_count + 1 + self.class_number),
                              H * W).permute(0, 2, 1).contiguous()

        x_point = []
        y_point = []
        x_point.append(torch.sigmoid(output[:, :, 0]))
        y_point.append(torch.sigmoid(output[:, :, 1]))
        for index in range(2, self.point_count, 2):
            x_point.append(output[:, :, index])
            y_point.append(output[:, :, index + 1])
        conf = torch.sigmoid(output[:, :, self.loc_count]).view(N, -1, 1)
        cls = output[:, :, self.loc_count + 1:self.loc_count + 1 + self.class_number].\
            view(N, -1, self.class_number)

        pred_corners = self.decode_predict_points(x_point, y_point, self.point_count,
                                                  N, H, W, device)
        pred_corners = pred_corners.transpose(0, 1).contiguous().\
            view(N, -1, self.loc_count)

        if targets is None:
            cls = F.softmax(cls, 2)
            return torch.cat([pred_corners, conf, cls], 2)
        else:
            num_ground_truth, num_det_correct, \
            coord_mask, conf_mask, cls_mask, \
            txs, tys, tconf, tcls = self.build_targets(pred_corners, targets, H, W, device)

            # conf
            conf = conf.view(N, H * W)
            conf_mask = conf_mask.sqrt()
            # cls
            cls = cls.view(-1, self.class_number)
            tcls = tcls[cls_mask].view(-1).long()
            cls_mask = cls_mask.view(-1, 1).repeat(1, self.class_number)
            cls = cls[cls_mask].view(-1, self.class_number)
            # Create loss
            loss_x = 0
            loss_y = 0
            for i in range(self.num_keypoints):
                loss_x = loss_x + self.coord_scale * self.mse_loss(x_point[i] * coord_mask,
                                                                   txs[i] * coord_mask)
                loss_y = loss_y + self.coord_scale * self.mse_loss(y_point[i] * coord_mask,
                                                                   tys[i] * coord_mask)
            loss_conf = self.mse_loss(conf * conf_mask, tconf * conf_mask) / 2.0
            loss_cls = self.class_scale * self.ce_loss(cls, tcls)

            loss = loss_x + loss_y + loss_conf + loss_cls

            self.info['num_ground_truth'] = num_ground_truth
            self.info['num_det_correct'] = num_det_correct
            self.info['x_loss'] = loss_x.item()
            self.info['y_loss'] = loss_y.item()
            self.info['conf_loss'] = loss_conf.item()
            self.info['cls_loss'] = loss_cls.item()
            self.print_info()
            return loss

