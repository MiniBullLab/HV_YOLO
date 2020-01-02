#python3 easyAI/main.py --task DeNET --gpu 0 --trainPath /home/lpj/github/data/Berkeley/ImageSets/train.txt --valPath /home/lpj/github/data/Berkeley/ImageSets/val.txt --cfg ./data/yolov3.cfg
python3 easyAI/main.py --task SegNET --gpu 0 --trainPath /home/lpj/github/data/LED_detect/ImageSets/train_val.txt --pretrainModel ./data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
