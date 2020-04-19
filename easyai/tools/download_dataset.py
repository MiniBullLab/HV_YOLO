#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from os import makedirs, remove
from six.moves import urllib
import tarfile


dataset_urls = {"BSDS300": "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"}


class DownloadDataset():

    def __init__(self):
        pass

    def download(self, url_path, output_dir):
        if not os.path.exists(output_dir):
            makedirs(output_dir)

        print("downloading url ", url_path)

        data = urllib.request.urlopen(url_path)
        data_name = os.path.basename(url_path)
        file_path = os.path.join(output_dir, data_name)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, output_dir)

        remove(file_path)
        return file_path


def main():
    print("start...")
    test = DownloadDataset()
    test.download(dataset_urls["BSDS300"], "/home/lpj/github/data/")
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()

