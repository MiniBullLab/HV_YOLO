#!/bin/bash
pid_file=.monitor.pid
host=$(cat /etc/hostname)
rm -rf ./.log/seg_logs/
service1="tensorboard --logdir=./.log/seg_logs/ --port=9999"
service2="google-chrome http://${host}:9999"

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
python3 -m easyAI.easy_ai_seg --gpu 0 --trainPath /home/minibull/lipeijie/dataset/LED_segment/ImageSets/train_val.txt
python3 -m easyAI.easy_convert --task SegNET
stop

set -v
root_path=$(pwd)
modelDir="./.log/model"
imageDir="./.log/seg_img"
outDir="${root_path}/.log/out"
tensorflowName=seg_best
outNetName=seg_best

inputColorFormat=1
outputShape=1,3,512,440
inputLayerName="i:net_input|is:1,512,440,3"
compare_inputLayerName="i:net_input=${imageDir}/img_list.txt|iq|idf:0,0,0,0|is:1,512,440,3"
temp_outputLayerName="conv2d_10/Sigmoid"
outputLayerName="o:conv2d_10/Sigmoid|ot:0,3,1,2|odf:fp32"
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
graph_surgery.py tf -p $modelDir/$tensorflowName.pb \
                    -o $modelDir/${tensorflowName}_temp.pb \
                    -isrc $inputLayerName \
                    -on $temp_outputLayerName \
                    -t ConstantifyShapes

CUDA_VISIBLE_DEVICES=-1
tfparser.py -p $modelDir/${tensorflowName}_temp.pb \
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

python3 -m easyAI.easy_encrypt -i $outDir/cavalry/$outNetName.bin -o ${root_path}/${outNetName}.bin
