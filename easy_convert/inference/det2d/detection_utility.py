#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
import math 
import cv2
from easy_convert.inference.utility.data_type import Point
from easy_convert.inference.utility.data_type import Rectangle
from easy_convert.inference.utility.data_type import ObjectBox
from functools import cmp_to_key


color_list = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]]


def logistic_activate(x):
    return 1. / (1. + math.exp(-x))


def get_region_box(x, biases, n, index, i, j, lw, lh, w, h, stride, mask):
    # print(n, biases[2 * mask[n]], biases[2 * mask[n] + 1])
    rect = Rectangle(Point((i + x[index + 0*stride]) / lw,
                           (j + x[index + 1*stride]) / lh),
                     math.exp(x[index + 2*stride]) * biases[2 * mask[n]] / w,
                     math.exp(x[index + 3*stride]) * biases[2 * mask[n] + 1] / h)
    box = ObjectBox(rect)
    return box


def compareScore(box1, box2):
    if box1.objectness < box2.objectness:
        return 1
    elif box1.objectness == box2.objectness:
        return 0
    else:
        return -1


def entry_index(lw, lh, classes, location, entry):
    n=location//(lw*lh)
    loc=location%(lw*lh)
    return n*lw*lh*(4+classes+1)+entry*lw*lh+loc


def do_nms_obj(boxes,total,classes,thresh):
    k=total-1
    for i in range(k):
        if(boxes[i].objectness==0):
            box_swap = boxes[i]
            boxes[i] = boxes[k]
            boxes[k] = box_swap
            k=k-1
            i=i-1
    boxes.sort(key=cmp_to_key(compareScore))
    for i in range(total):
        if boxes[i].objectness == 0:
            continue
        for j in range(i+1,total):
            if boxes[j].objectness == 0:
                continue
            if boxes[i].iou(boxes[j]) > thresh:
                boxes[j].objectness = 0
                for k in range(classes):
                    boxes[j].prob[k]=0


def detectYolo(boxes, count, classes,nms):
    do_nms_obj(boxes,count,classes,nms)
   
    results = []
    for j in range(count):
        for i in range(classes):
            if boxes[j].prob[i] > 0:
                # print("box info: {}, {}, {}, {}, {}, {}, {}".format(j, i, boxes[j].prob[i],boxes[j].rect.corner.x, boxes[j].rect.corner.y, boxes[j].rect.width, boxes[j].rect.height))
                results.append((i, boxes[j].prob[i], (boxes[j].rect.corner.x, boxes[j].rect.corner.y, boxes[j].rect.width, boxes[j].rect.height)))
    results = sorted(results, key=lambda x: -x[1])
    return results


def get_yolo_detections(feat, lw, lh, biases, boxes_of_each_grid,
                        classes, w, h, netw, neth, thresh, mask, relative):
    boxes = []
    channel, height, width = feat.shape
    predictions = forward_yolo(lw, lh, boxes_of_each_grid, classes, feat)
    count = 0
    # if mask[0] == 0:
    #     for k in range(0, lw * lh * 33):
    #         print("predictions: {} {}".format(k, predictions[k]))
    for i in range(lw * lh):
        row = i // width
        col = i % width
        for n in range(boxes_of_each_grid):
            obj_index = entry_index(lw, lh, classes, n*lw*lh + i, 4)
            scale = predictions[obj_index]
            # print("i: {}, n: {}, obj_index: {}, objectness: {}".format(i,n,obj_index,scale))
            if scale <= thresh:
                continue
            # print("i: {}, n: {}, obj_index: {}, objectness: {}".format(i,n,obj_index,scale))
            box_index = entry_index(lw, lh, classes, n*lw*lh + i, 0)
            box_tmp = get_region_box(predictions, biases, n, box_index, col, row, lw, lh, netw, neth, lw*lh, mask)
            # print("box_index: {}, box.x: {}, box.y: {}, box.w: {}, box.h: {}".format(box_index,box_tmp.rect.corner.x,box_tmp.rect.corner.y,box_tmp.rect.width,box_tmp.rect.height))
            probList = np.zeros(classes)
            for j in range(classes):
                class_index = entry_index(lw, lh, classes, n*lw*lh + i, 4 + 1 + j)
                prob = scale * predictions[class_index]
                # print "class_index: {} ,prob : {}".format(class_index, prob)
                probList[j] = prob if prob > thresh else 0
                # if prob > thresh:
		        # print("scale: {}, predictions: {}, prob_res: {}".format(scale,predictions[class_index], prob))
                #    probList[j] = prob
                # else:
                #    probList[j] = 0
            box_new = ObjectBox(box_tmp.rect,probList,scale)
            boxes.append(box_new)
            count = count+1

    correct_yolo_boxes(boxes, count, w, h, netw, neth, relative)

    return boxes, count


def correct_yolo_boxes(boxes, n, w, h, netw, neth, relative):#netw,neth need be float
    if(float(netw)/float(w) < float(neth)/float(h)):
        new_w=netw
        new_h=(h*netw)/w
    else:
        new_h=neth
        new_w=(w*neth)/h
    for i in range(n):
        box_x = (boxes[i].rect.corner.x - (netw - new_w)/2./netw) / (float(new_w)/float(netw))
        box_y = (boxes[i].rect.corner.y - (neth - new_h)/2./neth) / (float(new_h)/float(neth))
        box_w = boxes[i].rect.width*float(netw)/float(new_w)
        box_h = boxes[i].rect.height*float(neth)/float(new_h)
        # print("i: {}, box_x: {}, box_y: {}, box_w: {}, box_h: {}".format(i,box_x,box_y,box_w,box_h))
        if 1:
            box_x = box_x*w
            box_w = box_w*w
            box_y = box_y*h
            box_h = box_h*h
        boxes[i].rect.corner.x=box_x
        boxes[i].rect.corner.y=box_y
        boxes[i].rect.width=box_w
        boxes[i].rect.height=box_h


def forward_yolo(lw, lh, n, classes, feat):
    predictions = feat.reshape(-1)
    for i in range(n):
        index=entry_index(lw, lh, classes, i*lw*lh, 0)
        for j in range(2*lw*lh):
            predictions[index+j] = logistic_activate(predictions[index+j])
        index=entry_index(lw, lh, classes, i*lw*lh, 4)
        for j in range((1+classes)*lw*lh):
            predictions[index+j] = logistic_activate(predictions[index+j])

    return predictions


def draw_image(pic_name, results, name_list, scale):
    img_name = pic_name.split('/')[-1]
    im = cv2.imread(pic_name)
    height, width = im.shape[:2]
    for i in range(len(results)):
        rect = results[i][2]
        xmin = int(rect[0]-rect[2]/2.0)
        ymin = int(rect[1]-rect[3]/2.0)
        xmax = int(rect[0]+rect[2]/2.0)
        ymax = int(rect[1]+rect[3]/2.0)
        if xmin <= 0:
            xmin = 1
        if ymin <= 0:
            ymin = 1
        if xmax >= width:
            xmax = width - 1
        if ymax >= height:
            xmax = height - 1
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color_list[results[i][0]], 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, "{}:{:.1f}%".format(name_list[results[i][0]], results[i][1]*100), (xmin,ymin-10), font, 0.5, color_list[results[i][0]], 2, False)
    cv2.namedWindow("image", 0)
    cv2.resizeWindow("image", int(im.shape[1]*scale), int(im.shape[0]*scale))
    cv2.imshow("image", im)
    # cv.imwrite('result/'+img_name, im)
    print("store result/{}, finished!".format(img_name))
