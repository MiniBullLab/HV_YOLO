#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import codecs
import json
from easyai.helper.dataType import *


class JsonProcess():

    MIN_WIDTH = 0
    MIN_HEIGHT = 0

    def __init__(self):
        pass

    def parse_rect_data(self, json_path):
        if not os.path.exists(json_path):
            print("error:%s file not exists" % json_path)
            return
        with codecs.open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        image_name = data_dict['filename']
        objects_dict = data_dict['objects']
        rect_objects_list = objects_dict['rectObject']
        boxes = []
        for rect_dict in rect_objects_list:
            class_name = rect_dict['class']
            xmin = rect_dict['minX']
            ymin = rect_dict['minY']
            xmax = rect_dict['maxX']
            ymax = rect_dict['maxY']
            box = Rect2D()
            box.min_corner.x = xmin
            box.min_corner.y = ymin
            box.max_corner.x = xmax
            box.max_corner.y = ymax
            box.name = class_name
            if box.width() >= JsonProcess.MIN_WIDTH \
                    and box.height() >= JsonProcess.MIN_HEIGHT:
                boxes.append(box)
        return image_name, boxes

    def parse_key_points_data(self, json_path):
        if not os.path.exists(json_path):
            print("error:%s file not exists" % json_path)
            return
        with codecs.open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        image_name = data_dict['filename']
        objects_dict = data_dict['objects']
        rect_objects_list = objects_dict['rectObject']
        boxes = []
        for rect_dict in rect_objects_list:
            class_name = rect_dict['class']
            xmin = rect_dict['minX']
            ymin = rect_dict['minY']
            xmax = rect_dict['maxX']
            ymax = rect_dict['maxY']
            point_count = rect_dict['pointCount']
            key_points_list = rect_dict['keyPoints']
            box = Rect2D()
            box.min_corner.x = xmin
            box.min_corner.y = ymin
            box.max_corner.x = xmax
            box.max_corner.y = ymax
            box.name = class_name
            box.clear_key_points()
            for index in range(0, point_count, 2):
                point = Point2d(int(key_points_list[index]), int(key_points_list[index+1]))
                box.add_key_points(point)
            boxes.append(box)
        return image_name, boxes
