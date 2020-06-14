#!/bin/bash

set -v
root_path=$(pwd)
modelDir="./log/snapshot"
imageDir="./log/det3d_img"
outDir="${root_path}/log/out"
caffeNetName=yolo-pose_san
outNetName=3DNet

inputColorFormat=0
outputShape=1,3,480,640
outputLayerName="o:layer31-conv|odf:fp32"
inputDataFormat=0,0,8,0

mean=0.0
scale=255.0

rm -rf $outDir
mkdir $outDir
mkdir $outDir/dra_image_bin

#amba
source /usr/local/amba-cv-tools-2.1.7-20190815.ubuntu-16.04/env/cv22.env

#cuda10
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#caffe
export PYTHONPATH=/home/lpj/Software/caffe_amba_python3/python:$PYTHONPATH

ls $imageDir/*.* > $imageDir/img_list.txt

imgtobin.py -i $imageDir/img_list.txt \
            -o $outDir/dra_image_bin \
            -c $inputColorFormat \
            -d 0,0,0,0 \
            -s $outputShape

ls $outDir/dra_image_bin/*.bin > $outDir/dra_image_bin/dra_bin_list.txt

caffeparser.py -p $modelDir/$caffeNetName.prototxt \
               -m $modelDir/$caffeNetName.caffemodel \
               -i $outDir/dra_image_bin/dra_bin_list.txt \
               -iq -idf $inputDataFormat \
               -o $outNetName \
               -of $outDir/out_parser \
               -odst $outputLayerName \
               -cp ./custom_nodes.py \
               -cd ./custom_node.so 

cd $outDir/out_parser;vas -auto -show-progress $outNetName.vas

rm -rf ${outDir}/cavalry
mkdir -p ${outDir}/cavalry

cavalry_gen -d $outDir/out_parser/vas_output/ \
            -f $outDir/cavalry/$outNetName.bin \
            -p $outDir/ \
            -v > $outDir/cavalry/cavalry_info.txt

rm -rf vas_output

cp $outDir/cavalry/$outNetName.bin  ${root_path}/${outNetName}.bin
# python3 -m easyAI.easy_encrypt -i $outDir/cavalry/$outNetName.bin -o ${root_path}/${outNetName}.bin
