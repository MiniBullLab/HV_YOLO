#!/bin/bash

#cuda10
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 python3 -m easyai.tools.offline -t classify -f 2 -i /home/lpj/github/data/animals_detect/ImageSets/val.txt -a ./cls_result.txt -m classnet -w ./log/snapshot/cls_best.pt
