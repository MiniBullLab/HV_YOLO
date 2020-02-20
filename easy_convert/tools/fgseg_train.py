#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
import numpy as np
import tensorflow as tf
import random as rn

cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                              gpu_options=tf.GPUOptions(allow_growth=True))
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
#                               gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))

from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import keras
from keras.preprocessing import image
from sklearn.utils import compute_class_weight
import matplotlib.pyplot as plt
from easy_convert.helper.dirProcess import DirProcess
from easy_convert.keras_models.seg.fgsegnetv2 import FgSegNetV2
from easy_convert.keras_models.seg.my_fgsegnetv2 import MyFgSegNetV2


class FgSegV2Train():

    def __init__(self):
        self.dirProcess = DirProcess()
        self.annotation_post = ".png"
        self.lr = 0.01
        self.val_split = 0.2
        self.max_epoch = 100
        self.batch_size = 1

        if not os.path.exists("tests"):
            os.makedirs("tests")
        self.mdl_path = os.path.join("tests", 'mdl')

    def train(self, train_val_path, vgg_weights_path):

        data = self.get_train_data(train_val_path)
        img_shape = data[0][0].shape  # (height, width, channel)
        # model = FgSegNetV2(self.lr, img_shape, vgg_weights_path)
        model = MyFgSegNetV2(self.lr, img_shape, vgg_weights_path)
        model = model.init_model()

        # make sure that training input shape equals to model output
        input_shape = (img_shape[0], img_shape[1])
        output_shape = (model.output._keras_shape[1], model.output._keras_shape[2])
        assert input_shape == output_shape, 'Given input shape:' + str(
            input_shape) + ', but your model outputs shape:' + str(output_shape)

        checkpoints = keras.callbacks.ModelCheckpoint(self.mdl_path + '-weights-{epoch:02d}-{val_acc:.5f}-{val_loss:.5f}.h5',
                                                      monitor='val_loss', verbose=0, save_best_only=False,
                                                      save_weights_only=False, mode='auto', period=1)
        early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
        redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto')
        model.fit(data[0], data[1],
                  validation_split=self.val_split,
                  epochs=self.max_epoch, batch_size=self.batch_size,
                  callbacks=[redu, early, checkpoints], verbose=1, class_weight=data[2],
                  shuffle=True)
        del model, data, early, redu

    def get_train_data(self, train_path):
        train_datas = self.get_image_and_label_list(train_path)
        images = []
        labels = []
        for image_path, label_path in train_datas:
            x = image.load_img(image_path)
            x = image.img_to_array(x)
            # x /= 255.0
            images.append(x)

            y = image.load_img(label_path, grayscale=True)
            y = 255 - image.img_to_array(y)
            y /= 255.0
            y = np.floor(y)
            labels.append(y)

        images = np.asarray(images)
        labels = np.asarray(labels)

        # Shuffle the training data
        idx = list(range(images.shape[0]))
        np.random.shuffle(idx)
        np.random.shuffle(idx)
        images = images[idx]
        labels = labels[idx]

        # compute class weights
        cls_weight_list = []
        for i in range(labels.shape[0]):
            y = labels[i].reshape(-1)
            lb = np.unique(y)  # 0., 1
            cls_weight = compute_class_weight('balanced', lb, y)
            class_0 = cls_weight[0]
            class_1 = cls_weight[1] if len(lb) > 1 else 1.0

            cls_weight_dict = {0: class_0, 1: class_1}
            cls_weight_list.append(cls_weight_dict)
        del y
        cls_weight_list = np.asarray(cls_weight_list)

        return [images, labels, cls_weight_list]

    def get_image_and_label_list(self, train_path):
        result = []
        path, _ = os.path.split(train_path)
        images_dir = os.path.join(path, "../JPEGImages")
        labels_dir = os.path.join(path, "../SegmentLabel")
        for filename_and_post in self.dirProcess.getFileData(train_path):
            filename, post = os.path.splitext(filename_and_post)
            label_filename = filename + self.annotation_post
            label_path = os.path.join(labels_dir, label_filename)
            image_path = os.path.join(images_dir, filename_and_post)
            # print(image_path)
            if os.path.exists(label_path) and \
                    os.path.exists(image_path):
                result.append((image_path, label_path))
            else:
                print("%s or %s not exist" % (label_path, image_path))
        return result

    def show(self, result):
        plt.imshow(result)
        plt.title('Segmentation mask')
        plt.axis('off')
        plt.show()


def main():
    print("process start...")
    train_process = FgSegV2Train()
    train_process.train("/home/lpj/github/data/LED_detect/ImageSets/train_val.txt",
                        "./data/keras/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    print("process end!")


if __name__ == "__main__":
    main()
