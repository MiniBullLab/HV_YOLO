# -*- coding: utf-8 -*-

class Point2d():
    """
    Helper class to define Box
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        return

class DetectObject():

    def __init__(self):
        self.name = ""
        self.difficult = 0

class Rect2D(DetectObject):
    """
    class to define Box
    """

    def __init__(self):
        super().__init__()
        self.min_corner = Point2d(0, 0)
        self.max_corner = Point2d(0, 0)
        return

    def copy(self):
        b = Rect2D()
        b.min_corner = self.min_corner
        b.max_corner = self.max_corner
        return b

    def width(self):
        return self.max_corner.x - self.min_corner.x

    def height(self):
        return self.max_corner.y - self.min_corner.y

    def getVector(self):
        return [self.min_corner.x, self.min_corner.y,\
                self.max_corner.x, self.max_corner.y]

    def __str__(self):
        return '%s:%d %d %d %d' % (self.name, self.min_corner.x, self.min_corner.y, self.max_corner.x, self.max_corner.y)

