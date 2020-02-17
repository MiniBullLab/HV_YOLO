#!/bin/bash
pid_file=.monitor.pid
host=$(cat /etc/hostname)
rm -rf ./.log/det_logs
service1="tensorboard --logdir=./.log/det_logs --port=9998"
service2="google-chrome http://${host}:9998"

function start() {
    ${service1} &
    if [[ $? -eq 0 ]]; then
        echo $! > ${pid_file}
    else exit 1
    fi
    ${service2} &
    if [[ $? -eq 0 ]]; then
        echo $! >> ${pid_file}
    else exit 1
    fi
}

function stop() {
    # shellcheck disable=SC2046
    kill -9 $(cat ${pid_file})
    # shellcheck disable=SC2181
    if [[ $? -eq 0 ]]; then
        rm -f ${pid_file}
    else exit 1
    fi
}

start
python3 -m easyAI.easy_ai_det --gpu 0 --trainPath /home/minibull/lipeijie/dataset/Berkeley/ImageSets/train.txt --valPath /home/minibull/lipeijie/dataset/Berkeley/ImageSets/val.txt
python3 -m easyAI.easy_convert --task DeNET
stop

set -v
root_path=$(pwd)
modelDir="./.log/model"
imageDir="./.log/det_img"
outDir="${root_path}/.log/out"
caffeNetName=detection
outNetName=detection

inputColorFormat=0
outputShape=1,3,352,640
outputLayerName="o:628|odf:fp32"
outputLayerName1="o:654|odf:fp32"
outputLayerName2="o:680|odf:fp32"
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

caffeparser.py -p $modelDir/$caffeNetName.prototxt \
               -m $modelDir/$caffeNetName.caffemodel \
               -i $outDir/dra_image_bin/dra_bin_list.txt \
               -o $outNetName \
               -of $outDir/out_parser \
               -im $mean -ic $scale  -it 0,1,2,3 \
               -iq -idf $inputDataFormat -odst $outputLayerName -odst $outputLayerName1 -odst $outputLayerName2 # -c act-force-fx16,coeff-force-fx16 

cd $outDir/out_parser;vas -auto -show-progress $outNetName.vas

rm -rf ${outDir}/cavalry
mkdir -p ${outDir}/cavalry

cavalry_gen -d $outDir/out_parser/vas_output/ \
            -f $outDir/cavalry/$outNetName.bin \
            -p $outDir/ \
            -v > $outDir/cavalry/cavalry_info.txt

rm -rf vas_output

python3 -m easyAI.easy_encrypt -i $outDir/cavalry/$outNetName.bin -o ${root_path}/${outNetName}.bin
