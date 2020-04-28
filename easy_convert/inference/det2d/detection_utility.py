import os
import numpy as np
import math 
import cv2 as cv


class Point(object):
    def __init__(self, x=0., y=0.):
        self.x = x
        self.y = y

    def __str__(self):
        return "({},{})".format(self.x, self.y)


class Rectangle(object):
    def __init__(self, posn, w, h):
        self.corner = posn
        self.width = w
        self.height = h

    def __str__(self):
        return "({0},{1},{2})".format(self.corner, self.width, self.height)

    def iou(self, rect):
        return self.intersection(rect) / self.union(rect)

    def intersection(self, rect):
        w = overlap(self.corner.x, self.width, rect.corner.x, rect.width)
        h = overlap(self.corner.y, self.height, rect.corner.y, rect.height)
        if w < 0 or h < 0:
            return 0
        area = w * h
        return area

    def union(self, rect):
        i = self.intersection(rect)
        u = self.width * self.height + rect.width * rect.height - i
        return u


class Box(object):
    def __init__(self, rect, prob=np.zeros(100), objectness=-1):
        self.rect = rect
        self.prob = prob
        self.objectness = objectness

    def __str__(self):
        return "({0},{1},{2})".format(self.rect, self.prob, self.objectness)

    def iou(self, box2):
        return self.rect.iou(box2.rect)


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    right = r1 if r1 < r2 else r2
    return right - left


def logistic_activate(x):
    return 1. / (1. + math.exp(-x))


def get_region_box(x, biases, n, index, i, j, lw, lh, w, h, stride, mask):
    rect = Rectangle(Point((i + x[index + 0*stride]) / lw,
                           (j + x[index + 1*stride]) / lh),
                     math.exp(x[index + 2*stride]) * biases[2 * mask[n]] / w,
                     math.exp(x[index + 3*stride]) * biases[2 * mask[n] + 1] / h)
    box = Box(rect)
    return box


def compareScore(box1, box2):
    if box1.objectness < box2.objectness:
        return 1
    elif box1.objectness == box2.objectness:
        return 0
    else:
        return -1


def entry_index(lw, lh, classes, location, entry):
    n=location/(lw*lh)
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
    boxes.sort(cmp=compareScore)
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
                #print("box info: {}, {}, {}, {}, {}, {}, {}".format(j, i, boxes[j].prob[i],boxes[j].rect.corner.x, boxes[j].rect.corner.y, boxes[j].rect.width, boxes[j].rect.height))
            	results.append((i, boxes[j].prob[i], (boxes[j].rect.corner.x, boxes[j].rect.corner.y, boxes[j].rect.width, boxes[j].rect.height)))
    results = sorted(results, key=lambda x: -x[1])
    return results


def get_yolo_detections(feat, lw, lh, biases, boxes_of_each_grid, classes, w, h, netw, neth, thresh, mask, relative):
    boxes = []
    channel, height, width = feat.shape
    predictions = forward_yolo(lw, lh, boxes_of_each_grid, classes, feat)
    count=0
    #if mask[0] == 0:
    #	for k in xrange(lw * lh * 33):
    #    	print("predictions: {} {}".format(k, predictions[k]))
    for i in range(lw * lh):
        row = i / width
        col = i % width
        for n in range(boxes_of_each_grid):
            obj_index = entry_index(lw, lh, classes, n*lw*lh + i, 4)
            scale = predictions[obj_index]
            #print("i: {}, n: {}, obj_index: {}, objectness: {}".format(i,n,obj_index,scale))
            if scale <= thresh:
                continue
            #print("i: {}, n: {}, obj_index: {}, objectness: {}".format(i,n,obj_index,scale))
            box_index = entry_index(lw, lh, classes, n*lw*lh + i, 0)
            box_tmp = get_region_box(predictions, biases, n, box_index, col, row, lw, lh, netw, neth, lw*lh, mask)
            #print("box_index: {}, box.x: {}, box.y: {}, box.w: {}, box.h: {}".format(box_index,box_tmp.rect.corner.x,box_tmp.rect.corner.y,box_tmp.rect.width,box_tmp.rect.height))
            probList = np.zeros(classes)
            for j in range(classes):
                class_index = entry_index(lw, lh, classes, n*lw*lh + i, 4 + 1 + j)
                prob = scale * predictions[class_index]
                #print "class_index: {} ,prob : {}".format(class_index, prob)
                probList[j] = prob if prob > thresh else 0
                #if prob > thresh:
		        #print("scale: {}, predictions: {}, prob_res: {}".format(scale,predictions[class_index], prob))
                #    probList[j] = prob
                #else:
                #    probList[j] = 0
            box_new = Box(box_tmp.rect,probList,scale)
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
        #print("i: {}, box_x: {}, box_y: {}, box_w: {}, box_h: {}".format(i,box_x,box_y,box_w,box_h))
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


def preProcess(pic_name, width, height):
    img = cv.imread(pic_name)
    img_shape = [img.shape[0], img.shape[1]]
    rgbImage, ratio, padw, padh = resize_square(img,
                                                (width, height),
                                                (127.5, 127.5, 127.5))
    newr = rgbImage
    rgbImage = np.ascontiguousarray(rgbImage, dtype=np.float32)
    rgbImage /= 255.0
    in_ = rgbImage[:,:,::-1]
    in_ = in_.transpose((2,0,1))
    return newr, in_, img_shape


def resize_square(img, imageSize, color=(0, 0, 0)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(imageSize[0]) / shape[1], float(imageSize[1]) / shape[0])  # ratio  = old / new
    new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]
    dw = imageSize[0] - new_shape[1]  # width padding
    dh = imageSize[1] - new_shape[0]  # height padding
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv.resize(img, (int(new_shape[1]), int(new_shape[0])), interpolation=cv.INTER_AREA)  # resized, no border
    return cv.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv.BORDER_CONSTANT, value=color), ratio, dw, dh


def get_names_from_file(filename):
    result = []
    fd = file(filename, 'r')
    for line in fd.readlines():
        result.append(line.replace('\n', ''))
    return result


def get_color_from_file(filename):
    colors = []
    fd = file(filename, 'r')
    for line in fd.readlines():
        words = line.split(r',')
        color = (int(words[0]), int(words[1]), int(words[2]))
        colors.append(color)
    return colors


def draw_image(pic_name, results, name_list, scale):
    color_list = get_color_from_file('ink.color')
    img_name = pic_name.split('/')[-1]
    im = cv.imread(pic_name)
    height, width = im.shape[:2]
    for i in range(len(results)):
        rect = results[i][2]
        xmin = int(rect[0]-rect[2]/2.0)
        ymin = int(rect[1]-rect[3]/2.0)
        xmax = int(rect[0]+rect[2]/2.0)
        ymax = int(rect[1]+rect[3]/2.0)

        cv.rectangle(im, (xmin, ymin), (xmax, ymax), color_list[results[i][0]], 2)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(im, "{}:{:.1f}%".format(name_list[results[i][0]], results[i][1]*100), (xmin,ymin-10), font, 0.5, color_list[results[i][0]], 2, False)
    cv.namedWindow("image",0)
    cv.resizeWindow("image", int(im.shape[1]*scale), int(im.shape[0]*scale))
    cv.imshow("image", im)
    #cv.imwrite('result/'+img_name, im)
    print("store result/{}, finished!".format(img_name))
