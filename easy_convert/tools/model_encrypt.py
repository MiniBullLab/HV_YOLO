#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
from typing import Tuple
import codecs

# def random_key(length:int) -> int:
#     from secrets import token_bytes
#     key:bytes = token_bytes(nbytes=length)
#     key_int:int = int.from_bytes(key, 'big')
#     return key_int


class ModelEncrypt():

    def __init__(self):
        pass

    def encrypt_file(self, bin_path, output_path):
        key = self.__get_key()
        key_count = len(key)
        output_file = open(output_path, 'wb')
        with open(bin_path, 'rb') as file:
            data = file.read(key_count)
            data_count = len(data)
            while data_count > 0:
                if data_count == key_count:
                    output_data = self.encrypt(data, key)
                    output_file.write(output_data)
                else:
                    temp_key = key[0:data_count]
                    output_data = self.encrypt(data, temp_key)
                    output_file.write(output_data)
                data = file.read(key_count)
                data_count = len(data)

    def encrypt(self, data, key):
        data_int = int.from_bytes(data, 'big')
        key_int = int.from_bytes(key, 'big')
        output_int = data_int ^ key_int
        length = (output_int.bit_length() + 7) // 8
        output_data = int.to_bytes(output_int, length, 'big')
        return output_data

    def __get_key(self):
        key = 'dgsfhs1422'
        return key.encode('utf-8')


def main():
    print("process start...")
    process = ModelEncrypt()
    process.encrypt_file("", "")
    print("process end!")


if __name__ == "__main__":
    main()