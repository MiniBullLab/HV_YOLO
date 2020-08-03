#!/bin/bash

rm -rf ./.log/segment*
python3 -m easyai.easy_ai --task SegNET --gpu 0 --trainPath $1 --valPath $2

set -v
root_path=$(pwd)
modelDir="./.log/snapshot"
imageDir="./.log/seg_img"
outDir="${root_path}/.log/out"
modelName=segnet
outNetName=segnet

inputColorFormat=1
outputShape=1,3,400,500
outputLayerName="o:507|ot:0,1,2,3|odf:fp32"
inputDataFormat=0,0,0,0

mean=0.0
scale=255.0

rm -rf $outDir
mkdir $outDir
mkdir $outDir/dra_image_bin

#amba
source /usr/local/amba-cv-tools-2.1.7-20190815.ubuntu-18.04/env/cv22.env

#cuda10
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#caffe
export PYTHONPATH=/home/minibull/Software/caffe/python:$PYTHONPATH

ls $imageDir/*.* > $imageDir/img_list.txt
imgtobin.py -i $imageDir/img_list.txt \
            -o $outDir/dra_image_bin \
            -c $inputColorFormat \
            -d 0,0,0,0 \
            -s $outputShape

ls $outDir/dra_image_bin/*.bin > $outDir/dra_image_bin/dra_bin_list.txt

onnxparser.py -m $modelDir/${modelName}.onnx \
                -i $outDir/dra_image_bin/dra_bin_list.txt \
                -o $outNetName \
                -of $outDir/out_parser \
                -is $outputShape \
                -im $mean -ic $scale \
                -iq -idf $inputDataFormat \
                -odst $outputLayerName \
                -c act-allow-fp16,coeff-force-fx16

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
