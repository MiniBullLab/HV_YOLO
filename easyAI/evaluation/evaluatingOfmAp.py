import os.path
import numpy as np
from easyAI.helper import XMLProcess
from easyAI.helper import DirProcess
from easyAI.data_loader.dataLoader import DataLoader


class MeanApEvaluating():

    def __init__(self, valPath, className):
        path, _ = os.path.split(valPath)
        self.annotationDir = os.path.join(path, "../Annotations")
        self.xmlProcess = XMLProcess()
        self.dataLoader = DataLoader(path)
        self.className = className
        self.imageAndAnnotationList = self.dataLoader.getImageAndAnnotationList(valPath)

    def do_python_eval(self, output_dir, detection_path):
        detpath = detection_path  # the format and the address where the ./darknet detector valid .. results are stored
        aps = []
        ious = []
        # The PASCAL VOC metric changed in 2010

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self.className):
            filename = detpath + cls + '.txt'
            if cls == '__background__' :
                continue
            rec, prec, ap = self.voc_eval(filename, cls, ovthresh=0.5,
                                     use_07_metric=False)  # , avg_iou

            aps += [ap]
            # ious += [avg_iou]
            # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
            #     cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for i, ap in enumerate(aps):
            print(self.className[i] + ': ' + '{:.3f}'.format(ap))
            # print(self.className[i] + '_iou: ' + '{:.3f}'.format(ious[aps.index(ap)]))

        print('mAP: ' + '{:.3f}'.format(np.mean(aps)))
        # print('Iou acc: ' + '{:.3f}'.format(np.mean(ious)))
        print('~~~~~~~~')

        return np.mean(aps), aps

    def voc_eval(self, detpath,
                 classname,
                 ovthresh=0.5,
                 use_07_metric=False):
        if not os.path.exists(detpath):
            return 0, 0, 0
        recs = {}
        for imagePath, annotationPath in self.imageAndAnnotationList:
            path, fileNameAndPost = os.path.split(imagePath)
            fileName, post = os.path.splitext(fileNameAndPost)
            _, _, boxes = self.xmlProcess.parseRectData(annotationPath)
            recs[fileName] = boxes

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imageName in recs.keys():
            R = [box for box in recs[imageName] if box.name == classname]
            bbox = np.array([x.getVector() for x in R])
            difficult = np.array([x.difficult for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imageName] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

        # read dets
        with open(detpath, 'r') as f:
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

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        iou = []
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

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

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                        iou.append(ovmax)
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avg_iou = sum(iou) / len(iou)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec, use_07_metric)  #

        return rec, prec, ap  # , avg_iou

    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap