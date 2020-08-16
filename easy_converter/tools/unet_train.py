#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

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

from keras import backend as keras
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
keras.set_session(sess)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from easy_converter.keras_models.seg.unet import UNet
from easy_converter.keras_models.data_process.data_augment import trainGenerator, testGenerator, saveResult


data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(2, 'data/membrane/train', 'image','label',data_gen_args, save_to_dir=None)

unet = UNet()
model = unet.init_unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("log", results)

