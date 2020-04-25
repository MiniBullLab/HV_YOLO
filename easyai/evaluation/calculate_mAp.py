#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os.path
import numpy as np
from easyai.helper import XMLProcess
from easyai.data_loader.det.detection_sample import DetectionSample


class CalculateMeanAp():

    def __init__(self, val_path, class_names):
        self.class_names = class_names
        self.xmlProcess = XMLProcess()
        self.detection_samples = DetectionSample(val_path, class_names)
        self.image_annotation_list = self.detection_samples.get_image_and_label_list(val_path)
        self.use_07_metric = False

    def eval(self, result_dir):
        aps = []
        ious = []
        for i, name in enumerate(self.class_names):
            if name == '__background__':
                continue
            file_path = os.path.join(result_dir, "%s.txt" % name)
            recall, precision, ap = self.calculate_ap(file_path, name, 0.5)
            aps += [ap]
            # ious += [avg_iou]

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

    def calculate_ap(self, result_path, class_name, iou_thresh=0.5):
        if not os.path.exists(result_path):
            return 0, 0, 0

        recs = self.get_data_boxes()
        class_recs, npos = self.get_gt_boxes(recs, class_name)
        image_ids, sorted_scores, BB = self.get_detect_result(result_path)
        tp, fp, iou = self.get_tp_fp(image_ids, class_recs, BB, iou_thresh)

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

    def get_data_boxes(self):
        recs = {}
        for image_path, annotation_path in self.image_annotation_list:
            path, filename_post = os.path.split(image_path)
            #fileName, post = os.path.splitext(fileNameAndPost)
            _, _, boxes = self.xmlProcess.parseRectData(annotation_path)
            recs[filename_post] = boxes
        return recs

    def get_gt_boxes(self, recs, class_name):
        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imageName in recs.keys():
            R = [box for box in recs[imageName] if box.name == class_name]
            bbox = np.array([x.getVector() for x in R])
            difficult = np.array([x.difficult for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imageName] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det2d': det}
        return class_recs, npos

    def get_detect_result(self, result_path):
        # read dets
        with open(result_path, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        return image_ids, sorted_scores, BB

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
