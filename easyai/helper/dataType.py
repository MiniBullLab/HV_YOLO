#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


class Point2d():
    """
    Helper class to define 2dBox
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        return


class Point3d():
    """
    Helper class to define 3dBox
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        return


class MyObject():

    def __init__(self):
        self.name = ""
        self.class_id = -1
        self.difficult = 0


class Rect2D(MyObject):
    """
    class to define 3dBox
    """

    def __init__(self):
        super().__init__()
        self.min_corner = Point2d(0, 0)
        self.max_corner = Point2d(0, 0)
        self.key_points = []

    def copy(self):
        b = Rect2D()
        b.min_corner = self.min_corner
        b.max_corner = self.max_corner
        return b

    def center(self):
        x = (self.max_corner.x + self.min_corner.x) / 2
        y = (self.max_corner.y + self.min_corner.y) / 2
        return x, y

    def width(self):
        return self.max_corner.x - self.min_corner.x

    def height(self):
        return self.max_corner.y - self.min_corner.y

    def getTuple(self):
        return (self.min_corner.x, self.min_corner.y,
                self.max_corner.x, self.max_corner.y)

    def getVector(self):
        return [self.min_corner.x, self.min_corner.y,
                self.max_corner.x, self.max_corner.y]

    def clear_key_points(self):
        self.key_points.clear()

    def add_key_points(self, point):
        self.key_points.append(point)

    def get_key_points(self):
        return self.key_points

    def __str__(self):
        return '%s:%d %d %d %d' % (self.name, self.min_corner.x, self.min_corner.y, self.max_corner.x, self.max_corner.y)


class Rect3D(MyObject):
    """
    class to define 3dBox
    """

    def __init__(self):
        super().__init__()
        self.center_corner = Point3d(0, 0, 0)
        self.size = Point3d(0, 0, 0)


class DetectionObject(Rect2D):

    def __init__(self):
        super().__init__()
        self.classIndex = -1
        self.classConfidence = 0
        self.objectConfidence = 0

