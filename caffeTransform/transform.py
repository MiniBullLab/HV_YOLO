# import tensorflow as tf
# from collections import OrderedDict
# import numpy as np
#
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#
# from tensorflow.python import pywrap_tensorflow
# checkpoint_path = "./bdd-200000"
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
#
# new_state_dict = OrderedDict()
# for key_layer, val in var_to_shape_map.items():
#     if "encode" in key_layer and "Momentum" not in key_layer:
#         key_name = key_layer.split("/")
#         name = key_name[-3]+"_"+key_name[-1]  # remove `module.`
#         new_state_dict[name] = reader.get_tensor(key_layer)
#     if "decode" in key_layer:
#         key_name = key_layer.split("/")
#         name = key_name[-2]+"_"+key_name[-1]  # remove `module.`
#         new_state_dict[name] = reader.get_tensor(key_layer)
#
#torch.save(new_state_dict, "./fcn.pth")


# dict = ['conv1_1_W', 'conv1_1_moving_mean', 'conv1_1_moving_variance','conv1_1_gamma', 'conv1_1_beta',
# 'conv2_1_W', 'conv2_1_moving_mean', 'conv2_1_moving_variance','conv2_1_gamma', 'conv2_1_beta',
# 'conv3_1_W', 'conv3_1_moving_mean', 'conv3_1_moving_variance','conv3_1_gamma', 'conv3_1_beta',
# 'conv3_2_W', 'conv3_2_moving_mean', 'conv3_2_moving_variance','conv3_2_gamma', 'conv3_2_beta',
# 'conv4_1_W', 'conv4_1_moving_mean', 'conv4_1_moving_variance','conv4_1_gamma', 'conv4_1_beta',
# 'conv4_2_W', 'conv4_2_moving_mean', 'conv4_2_moving_variance','conv4_2_gamma', 'conv4_2_beta',
# 'conv4_3_W', 'conv4_3_moving_mean', 'conv4_3_moving_variance','conv4_3_gamma', 'conv4_3_beta',
# 'conv5_1_W', 'conv5_1_moving_mean', 'conv5_1_moving_variance','conv5_1_gamma', 'conv5_1_beta',
# 'conv5_2_W', 'conv5_2_moving_mean', 'conv5_2_moving_variance','conv5_2_gamma', 'conv5_2_beta',
# 'conv5_3_W', 'conv5_3_moving_mean', 'conv5_3_moving_variance','conv5_3_gamma', 'conv5_3_beta',
# 'score_origin_W', 'score_1_W', 'score_2_W', 'score_3_W', 'score_4_W', 'score_final_W']

import numpy as np
import caffe
import torch
from collections import OrderedDict

model = torch.load("./model8layers/model_epoch_100.pth")
new_state_dict = OrderedDict()

# for k in dict:
#     new_state_dict[k] = model[k]

# for k, v in model.state_dict().items():
#     print(k, v.shape)

vs = []
for index, (k, v) in enumerate(model.state_dict().items()):
    vs.append(v)
    print(str(index) + " " + k)

net = caffe.Net("./srNew.prototxt", caffe.TEST)
params = net.params.keys()

index = 0
for pr in params:
    lidx = list(net._layer_names).index(pr)
    layer = net.layers[lidx]
    # conv_bias = None
    if layer.type == 'Convolution':
        print(str(index) + " " + pr + "(conv)")
        # weights
        dims = net.params[pr][0].data.shape
        weightSize = np.prod(dims)
        net.params[pr][0].data[...] = np.reshape(vs[index], dims)
        if len(net.params[pr]) > 1:
            net.params[pr][1].data[...] = vs[index + 1]  # bias beta
            conv_bias = None
        index += 2
    elif layer.type == 'BatchNorm':
        print(str(index) + " " + pr + "(batchnorm)")
        net.params[pr][0].data[...] = vs[index + 1]  # mean
        net.params[pr][1].data[...] = vs[index + 2]  # variance
        net.params[pr][2].data[...] = 1.0  # scale factor
        index += 2
    elif layer.type == 'Scale':
        print(str(index) + " " + pr + "(scale)")
        net.params[pr][0].data[...] = vs[index]  # scale gamma
        batch_norm = None
        if len(net.params[pr]) > 1:
            net.params[pr][1].data[...] = vs[index + 1]  # bias beta
            conv_bias = None
        index += 2
    else:
        print("WARNING: unsupported layer, " + pr)

net.save("./srNew.caffemodel")
