import math

def Iou_cal(box1, box2):
    box1_rect = StrtoInt(box1)
    box2_rect = StrtoInt(box2)
    return float(intersection(box1_rect, box2_rect)) / float(union(box1_rect, box2_rect))

def StrtoInt(box):
    return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

def intersection(box1, box2):
    w = overlap(box1[0], box1[2], box2[0], box2[2])
    h = overlap(box1[1], box1[3], box2[1], box2[3])
    if w < 0 or h < 0:
        return 0
    area = w * h
    return area

def union(box1, box2):
    i = intersection(box1, box2)
    u = (box1[2]-box1[0]) * (box1[3]-box1[1]) + (box2[2]-box2[0]) * (box2[3]-box2[1]) - i
    return u

def overlap(x1, y1, x2, y2):
    left = x1 if x1 > x2 else x2
    right = y1 if y1 < y2 else y2
    return right - left
