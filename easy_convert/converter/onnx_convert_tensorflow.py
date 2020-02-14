import os
from optparse import OptionParser
import pathlib
import onnx
from onnx_tf import backend


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is onnx convert to tensorflow"

    parser.add_option("-i", "--input", dest="input_path",
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

    return options


class OnnxConvertTensorflow():

    def __init__(self, onnx_path):
        self.target_opset = 10
        self.onnx_path = pathlib.Path(onnx_path)
        self.tensorflow_model_save_path = self.onnx_path.with_suffix(".pb")

    def convert_tensorflow(self):
        model = onnx.load(str(self.onnx_path))
        tf_rep = backend.prepare(model, optset_version=self.target_opset)
        tf_rep.export_graph(str(self.tensorflow_model_save_path))

    def print_param(self, tf_rep):
        # Input nodes to the model
        print('inputs:', tf_rep.inputs)

        # Output nodes from the model
        print('outputs:', tf_rep.outputs)

        # All nodes in the model
        print('tensor_dict:')
        print(tf_rep.tensor_dict)


def main():
    print("process start...")
    options = parse_arguments()
    converter = OnnxConvertTensorflow(options.input_path)
    converter.convert_tensorflow()
    print("process end!")


if __name__ == "__main__":
    main()
