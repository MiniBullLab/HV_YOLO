#!/bin/bash

#cuda10
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#caffe
export PYTHONPATH=/opt/caffe/python:$PYTHONPATH

rm -rf ./log/detect2d*

CUDA_VISIBLE_DEVICES=0 python3 -m easy_tools.easy_ai --task DeNET --gpu 0 --trainPath $1 --valPath $2
python3 -m easy_tools.easy_convert --task DeNET --input ./log/snapshot/detnet.onnx

set -v
root_path=$(pwd)
modelDir="./log/snapshot"
imageDir="./log/det_img"
outDir="${root_path}/log/out"
caffeNetName=detnet
outNetName=detnet

inputColorFormat=0
outputShape=1,3,416,416
outputLayerName="o:636|odf:fp32"
outputLayerName1="o:662|odf:fp32"
outputLayerName2="o:688|odf:fp32"
inputDataFormat=0,0,8,0

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
export PYTHONPATH=/opt/caffe/python:$PYTHONPATH

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
               -o $outNetName \
               -of $outDir/out_parser \
               -it 0,1,2,3 \
               -iq -idf $inputDataFormat -odst $outputLayerName -odst $outputLayerName1 -odst $outputLayerName2 # -c act-force-fx16,coeff-force-fx16 

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
