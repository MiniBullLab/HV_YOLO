#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os.path
import numpy as np
from easyai.helper.dataType import DetectionObject
from easyai.helper.dirProcess import DirProcess
from easyai.helper.json_process import JsonProcess
from easyai.data_loader.det2d.det2d_sample import DetectionSample


class CalculateMeanAp():

    def __init__(self, class_names):
        self.class_names = class_names
        self.dir_process = DirProcess()
        self.json_process = JsonProcess()
        self.use_07_metric = False

    def eval(self, result_dir, val_path):
        aps = []
        for index, name in enumerate(self.class_names):
            file_path = os.path.join(result_dir, "%s.txt" % name)
            if not os.path.exists(file_path):
                aps.append(0)
            else:
                gt_boxes = self.get_gt_boxes(val_path, name)
                detect_boxes = self.get_detect_boxes(file_path)
                recall, precision, ap = self.calculate_ap(gt_boxes, detect_boxes, 0.5)
                aps += [ap]
        self.print_evaluation(aps)
        return np.mean(aps), aps

    def result_eval(self, result_path, val_path):
        aps = []
        for index, name in enumerate(self.class_names):
            gt_boxes = self.get_gt_boxes(val_path, name)
            detect_boxes = self.get_detect_boxes(result_path, name)
            if len(detect_boxes) == 0:
                aps.append(0)
            else:
                recall, precision, ap = self.calculate_ap(gt_boxes, detect_boxes, 0.5)
                aps += [ap]
        self.print_evaluation(aps)
        return np.mean(aps), aps

    def print_evaluation(self, aps):
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for i, ap in enumerate(aps):
            print(self.class_names[i] + ': ' + '{:.3f}'.format(ap))
            # print(self.className[i] + '_iou: ' + '{:.3f}'.format(ious[aps.index(ap)]))
        print('mAP: ' + '{:.3f}'.format(np.mean(aps)))
        # print('Iou acc: ' + '{:.3f}'.format(np.mean(ious)))
        print('~~~~~~~~')

    def get_gt_boxes(self, val_path, class_name):
        result = {}
        detection_samples = DetectionSample(val_path, self.class_names)
        image_annotation_list = detection_samples.get_image_and_label_list(val_path)
        for image_path, annotation_path in image_annotation_list:
            path, filename_post = os.path.split(image_path)
            _, boxes = self.json_process.parse_rect_data(annotation_path)
            result_boxes = [box for box in boxes if box.name == class_name]
            result[filename_post] = result_boxes
        return result

    def get_detect_boxes(self, result_path, class_name=None):
        result = []
        if class_name is None:
            for line_data in self.dir_process.getFileData(result_path):
                split_datas = [x.strip().split(' ') for x in line_data if x.strip()]
                filename_post = split_datas[0].strip()
                # print(filename_post)
                temp_object = DetectionObject()
                temp_object.objectConfidence = float(split_datas[1])
                temp_object.min_corner.x = int(split_datas[2])
                temp_object.min_corner.y = int(split_datas[3])
                temp_object.max_corner.x = int(split_datas[4])
                temp_object.max_corner.y = int(split_datas[5])
                result.append({filename_post, temp_object})
        else:
            for line_data in self.dir_process.getFileData(result_path):
                split_datas = [x.strip().split('|') for x in line_data if x.strip()]
                filename_post = split_datas[0].strip()
                # print(filename_post)
                for temp_box in split_datas[1:]:
                    box_datas = [x.strip().split(' ') for x in temp_box if x.strip()]
                    if box_datas[0] != class_name:
                        continue
                    temp_object = DetectionObject()
                    temp_object.objectConfidence = float(box_datas[1])
                    temp_object.min_corner.x = int(box_datas[2])
                    temp_object.min_corner.y = int(box_datas[3])
                    temp_object.max_corner.x = int(box_datas[4])
                    temp_object.max_corner.y = int(box_datas[5])
                    result.append({filename_post, temp_object})
        return result

    def calculate_ap(self, gt_boxes, detect_boxes, iou_thresh=0.5):
        class_recs, npos = self.process_gt_boxes(gt_boxes)
        image_ids, sorted_scores, boxes = self.process_detect_result(detect_boxes)
        tp, fp, iou = self.get_tp_fp(image_ids, class_recs, boxes, iou_thresh)

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        recall = tp / float(npos)
        # avg_iou = sum(iou) / len(iou)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.get_ap(recall, precision)
        return recall, precision, ap

    def process_gt_boxes(self, gt_boxes):
        class_recs = {}
        npos = 0
        for filename_post, boxes in gt_boxes.items():
            bbox = np.array([x.getVector() for x in boxes])
            difficult = np.array([x.difficult for x in boxes]).astype(np.bool)
            det = [False] * len(boxes)
            npos = npos + sum(~difficult)
            class_recs[filename_post] = {'bbox': bbox,
                                         'difficult': difficult,
                                         'det2d': det}
        return class_recs, npos

    def process_detect_result(self, detect_boxes):
        image_ids = []
        confidence = []
        boxes = []
        for image_id, temp_object in detect_boxes:
            image_ids.append(image_id)
            confidence.append(temp_object.objectConfidence)
            boxes.append([
                float(temp_object.min_corner.x),
                float(temp_object.min_corner.y),
                float(temp_object.max_corner.x),
                float(temp_object.max_corner.y)
            ])
        confidence = np.array(confidence)
        boxes = np.array(boxes)

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        boxes = boxes[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        return image_ids, sorted_scores, boxes

    def calculate_iou(self, BBGT, bb):
        ovmax = -np.inf
        jmax = None
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        return ovmax, jmax

    def get_tp_fp(self, image_ids, class_recs, BB, iou_thresh):
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        iou = []
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            BBGT = R['bbox'].astype(float)
            ovmax, jmax = self.calculate_iou(BBGT, bb)
            if ovmax > iou_thresh:
                if not R['difficult'][jmax]:
                    if not R['det2d'][jmax]:
                        tp[d] = 1.
                        R['det2d'][jmax] = 1
                        iou.append(ovmax)
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.
        return tp, fp, iou

    def get_ap(self, recall, precision):
        """
        ap = voc_ap(rec, prec, [use_07_metric])
                Compute VOC AP given precision and recall.
                If use_07_metric is true, uses the
                VOC 07 11 point method (default:False).
        """
        if self.use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], recall, [1.]))
            mpre = np.concatenate(([0.], precision, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
