#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import uuid


class ComputeDeviceProcess():

    def __init__(self):
        pass

    def get_mac_address(self):
        mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
        return ":".join([mac[e:e + 2] for e in range(0, 11, 2)])

