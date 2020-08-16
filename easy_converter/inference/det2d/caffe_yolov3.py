import numpy as np
import caffe
import cv2
from easy_converter.helper.imageProcess import ImageProcess
from easy_converter.helper.image_dataset_process import ImageDataSetProcess
from easy_converter.inference.det2d import detection_utility
from easy_converter.helper.dirProcess import DirProcess


class CaffeYoloV3Inference():

	def __init__(self, model_def, model_weights):
		self.dir_process = DirProcess()
		self.image_process = ImageProcess()
		self.dataset_process = ImageDataSetProcess()
		self.image_size = (640, 352)   # w, h
		self.output_node_1 = 'layer81-conv'
		self.output_node_2 = 'layer93-conv'
		self.output_node_3 = 'layer105-conv'
		self.class_list = ('truck', 'person', 'bicycle', 'car', 'motorbike', 'bus')
		self.classes = 6
		self.nms_threshold = 0.45
		self.thresh_conf = 0.24
		self.box_of_each_grid = 3
		self.biases = np.array([8, 9, 12, 29, 18, 15, 26, 64, 33, 26,
								61, 44, 58, 135, 150, 122, 208, 203])
		self.image_pad_color = (127.5, 127.5, 127.5)
		self.net = caffe.Net(model_def, model_weights, caffe.TEST)

	def yolov3_detect(self, image_dir):
		for img_path in self.dir_process.getDirFiles(image_dir):
			src_size, img = self.image_pre_process(img_path, self.image_size)
			self.net.blobs['data'].data[...] = img
			output = self.net.forward()
			feature_list = []
			feat1 = self.net.blobs[self.output_node_1].data[0]
			feature_list.append(feat1)
			feat2 = self.net.blobs[self.output_node_2].data[0]
			feature_list.append(feat2)
			feat3 = self.net.blobs[self.output_node_3].data[0]
			feature_list.append(feat3)

			totalBoxes = []
			totalCount = 0
			mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
			for index, feat in enumerate(feature_list):
				# n, c, h, w
				boxes, count = detection_utility.get_yolo_detections(feat, feat.shape[2],
																	feat.shape[1], self.biases,
																	self.box_of_each_grid, self.classes,
																	src_size[0], src_size[1],
																	self.image_size[0], self.image_size[1],
																	self.thresh_conf, mask[index], 0)

				totalBoxes += boxes
				totalCount += count
			results = detection_utility.detectYolo(totalBoxes, totalCount, self.classes, self.nms_threshold)
			detection_utility.draw_image(img_path, results, self.class_list, scale=0.8)
			key = cv2.waitKey()
			if key == 1048603 or key == 27:
				break

	def image_pre_process(self, image_path, input_size):
		_, src_image = self.image_process.readRgbImage(image_path)
		src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
		ratio, pad_size = self.dataset_process.get_square_size(src_size, input_size)
		image = self.dataset_process.image_resize_square(src_image, ratio, pad_size,
														color=self.image_pad_color)
		image = self.dataset_process.image_normaliza(image)
		image = self.dataset_process.numpy_transpose(image)
		return src_size, image


def main():
	caffe.set_device(0)
	caffe.set_mode_gpu()

	model_def = './data/darknet/yolov3_berkeley_6_classes.prototxt'
	model_weights = './data/darknet/yolov3_berkeley_6_classes.caffemodel'
	test = CaffeYoloV3Inference(model_def, model_weights)
	test.yolov3_detect("/home/lpj/Desktop/test")


if __name__ == "__main__":
	main()
