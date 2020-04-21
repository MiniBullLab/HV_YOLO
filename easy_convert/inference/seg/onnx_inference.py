import os
from optparse import OptionParser
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
import cv2


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is onnx inference"

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="PATH", type="string", default=None,
                      help="image path")

    parser.add_option("-o", "--onnx", dest="onnx_path",
                      metavar="PATH", type="string", default=None,
                      help="onnx path")

    (options, args) = parser.parse_args()

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
        else:
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input' option is required to run this program")

    if options.onnx_path:
        if not os.path.exists(options.onnx_path):
            parser.error("Could not find the onnx file")
        else:
            options.onnx_path = os.path.normpath(options.onnx_path)
    else:
        parser.error("'onnx' option is required to run this program")

    return options


class OnnxInference():

    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.session = onnxruntime.InferenceSession(self.onnx_path, None)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.output_size = (512, 440)

    def infer(self, data_path):
        input_data = cv2.imread(data_path)
        data = self.preprocess(input_data)
        raw_result = self.session.run([self.output_name], {self.input_name: data})
        probs = self.postprocess(raw_result)
        threshold = 0.8
        probs[probs < threshold] = 0.
        probs[probs >= threshold] = 255.
        self.show(probs)

    def preprocess(self, input_data):
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        result = np.array(input_data).astype('float32')
        result = np.expand_dims(result, axis=0)
        return result

    def postprocess(self, result):
        output_data = np.array(result).squeeze()
        output_data.reshape(self.output_size[0],
                            self.output_size[1])
        return output_data

    def softmax(self, x):
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def print_model_inputs(self):
        for x in self.session.get_inputs():
            print(x.name, x.shape)

    def print_model_outputs(self):
        for x in self.session.get_outputs():
            print(x.name, x.shape)

    def show(self, result):
        plt.subplot(1, 1, 1)
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(result)

        plt.title('Segmentation mask')
        plt.axis('off')
        plt.show()


def main():
    print("process start...")
    options = parse_arguments()
    inference = OnnxInference(options.onnx_path)
    inference.infer(options.input_path)
    print("process end!")


if __name__ == "__main__":
    main()
