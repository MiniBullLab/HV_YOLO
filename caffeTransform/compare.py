import cv2, torch
import caffe
from model import Net
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image

def postProcessing(out_img_y):
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    return out_img

def pixelS(input, scale):
    b, c, h, w = input.shape

    out_c = c / (scale * scale)
    rSquare = scale * scale
    output = np.zeros([b, int(out_c), int(h*scale), int(w*scale)])

    for bn in range(0, b):
        for k in range(0, c):
            for j in range(0, h):
                for i in range(0, w):
                    c2 = int(k / rSquare)
                    h2 = int(j * scale + ((k / scale) % scale))
                    w2 = int(i * scale + k % scale)
                    output[bn, c2, h2, w2] = input[bn, k, j, i]

    return output

# torch model
img = Image.open("000001.jpg").convert('YCbCr')
y, cb, cr = img.split()
img_to_tensor = ToTensor()

input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

model = torch.load("./model8layers/model_epoch_100.pth")

model = model.cuda()
input = input.cuda()

out = model(input)
out = out[0].detach().cpu().numpy()

out_img = postProcessing(out)
out_img.save("pyorch.png")

net = caffe.Net("./srNew.prototxt", "srNew.caffemodel", caffe.TEST)
params = net.params.keys()
#
# for pr in params:
#     print(pr)
input_ = input.cpu().numpy()[0]

net.blobs['data'].reshape(1, *input_.shape)
net.blobs['data'].data[...] = input_
net.forward()
prob = net.blobs['SRConv10'].data

outUPS = pixelS(prob, 3)[0]
prob_img = postProcessing(outUPS)
prob_img.save("caffe.png")

# minus_result = prob-out.detach().cpu().numpy()
# mse = np.sum(minus_result*minus_result)
#
# for i in range(0, 9):
#     plt.imsave("./pytorch" + str(i) + ".jpg", out[i].detach().cpu().numpy())
#     plt.imsave("./caffe" + str(i) + ".jpg", prob[i])
# caffe_model.blobs[input_name].data[...] = var_numpy
# net_output = caffe_model.forward()
# caffe_out = net_output[output_name]

# input_name = str(graph.inputs[0][0])
# output_name = str(graph.outputs[0][0])
#
# # get caffe output
# caffe_model.blobs[input_name].data[...] = var_numpy
# net_output = caffe_model.forward()
# caffe_out = net_output[output_name]
#
# # com mse between caffe and pytorch
# minus_result = caffe_out-pt_out
# mse = np.sum(minus_result*minus_result)