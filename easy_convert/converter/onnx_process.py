import os
import pathlib
from optparse import OptionParser
import onnx


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is onnx convert to onnx"

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


class OnnxProcess():

    def __init__(self, onnx_path):
        self.onnx_path = pathlib.Path(onnx_path)
        self.onnx_save_path = self.onnx_path.parent / \
                              pathlib.Path("%s_temp.onnx" % self.onnx_path.stem)
        self.endpoint_names = ['net_input:0', 'output:0']

    def convert(self):
        onnx_model = onnx.load(str(self.onnx_path))

        for i in range(len(onnx_model.graph.node)):
            for j in range(len(onnx_model.graph.node[i].input)):
                if onnx_model.graph.node[i].input[j] in self.endpoint_names:
                    print('-'*60)
                    print(onnx_model.graph.node[i].name)
                    print(onnx_model.graph.node[i].input)
                    print(onnx_model.graph.node[i].output)
                    onnx_model.graph.node[i].input[j] = onnx_model.graph.node[i].input[j].split(':')[0]

            for j in range(len(onnx_model.graph.node[i].output)):
                if onnx_model.graph.node[i].output[j] in self.endpoint_names:
                    print('-'*60)
                    print(onnx_model.graph.node[i].name)
                    print(onnx_model.graph.node[i].input)
                    print(onnx_model.graph.node[i].output)

                    onnx_model.graph.node[i].output[j] = onnx_model.graph.node[i].output[j].split(':')[0]

        for i in range(len(onnx_model.graph.input)):
            if onnx_model.graph.input[i].name in self.endpoint_names:
                print('-'*60)
                print(onnx_model.graph.input[i])
                onnx_model.graph.input[i].name = onnx_model.graph.input[i].name.split(':')[0]

        for i in range(len(onnx_model.graph.output)):
            if onnx_model.graph.output[i].name in self.endpoint_names:
                print('-'*60)
                print(onnx_model.graph.output[i])
                onnx_model.graph.output[i].name = onnx_model.graph.output[i].name.split(':')[0]

        onnx.save(onnx_model, str(self.onnx_save_path))


def main():
    print("process start...")
    options = parse_arguments()
    converter = OnnxProcess(options.input_path)
    converter.convert()
    print("process end!")


if __name__ == "__main__":
    main()
