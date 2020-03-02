import os
from optparse import OptionParser
import pathlib
import caffe
import onnx
from caffe.proto import caffe_pb2
from easyAI.converter.onnx2caffe.transformers import ConvAddFuser,ConstantsToInitializers
from easyAI.converter.onnx2caffe.graph import Graph
import easyAI.converter.onnx2caffe.operators as cvt
import easyAI.converter.onnx2caffe.weightloader as wlr
from easyAI.converter.onnx2caffe.error_utils import ErrorHandling
from onnx import shape_inference


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is onnx convert to caffe"

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="PATH", type="string", default=None,
                      help="onnx path")

    parser.add_option("-p", "--proto", dest="proto_path",
                      metavar="PATH", type="string", default=None,
                      help="prototxt path")

    (options, args) = parser.parse_args()

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
        else:
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input' option is required to run this program")

    return options


class OnnxConvertCaffe():

    transformers = [
        ConstantsToInitializers(),
        ConvAddFuser(),
    ]

    def __init__(self, onnx_path):
        caffe.set_mode_cpu()
        self.err = None
        self.onnx_path = pathlib.Path(onnx_path)
        self.proto_save_path = self.onnx_path.with_suffix(".prototxt")
        self.caffe_model_save_path = self.onnx_path.with_suffix(".caffemodel")
        self.graph = self.get_graph(onnx_path)
        self.layers = self.get_layers(self.graph)

    def convert_caffe(self):
        self.convert_proto(str(self.proto_save_path))
        self.convert_weights(str(self.proto_save_path))

    def convert_proto(self, proto_save_path):
        net = caffe_pb2.NetParameter()
        for index, layer in enumerate(self.layers):
            if index == 0:
                layer.layer_name = "data"
                layer.outputs = ["data"]
            elif index == 1:
                layer.inputs = ["data"]
            self.layers[index] = layer._to_proto()
        net.layer.extend(self.layers)
        with open(proto_save_path, 'w') as f:
            print(net, file=f)

    def convert_weights(self, proto_path):
        net = caffe.Net(proto_path, caffe.TEST)
        for id, node in enumerate(self.graph.nodes):
            node_name = node.name
            op_type = node.op_type
            inputs = node.inputs
            inputs_tensor = node.input_tensors
            input_non_exist_flag = False
            if op_type not in wlr._ONNX_NODE_REGISTRY:
                self.err.unsupported_op(node)
                continue
            converter_fn = wlr._ONNX_NODE_REGISTRY[op_type]
            converter_fn(net, node, self.graph, self.err)

        net.save(str(self.caffe_model_save_path))
        return net

    def convert_mse_error(self, caffe_net):
        pass
        # input_name = str(self.graph.inputs[0][0])
        # output_name = str(self.graph.outputs[0][0])
        #
        # # get caffe output
        # caffe_net.blobs[input_name].data[...] = var_numpy
        # net_output = caffe_net.forward()
        # caffe_out = net_output[output_name]
        #
        # # com mse between caffe and pytorch
        # minus_result = caffe_out-pt_out
        # mse = np.sum(minus_result*minus_result)
        #
        # print("{} mse between caffe and pytorch model output: {}".format(module_name,mse))

    def get_graph(self, onnx_path):
        model = onnx.load(onnx_path)
        model = shape_inference.infer_shapes(model)
        model_graph = model.graph
        graph = Graph.from_onnx(model_graph)
        #graph = graph.transformed(OnnxConvertCaffe.transformers)
        graph.channel_dims = {}
        for id, node in enumerate(graph.nodes):
            if node.op_type == "Upsample":
                del node.inputs[1]

        return graph

    def get_layers(self, graph):
        exist_edges = []
        layers = []
        exist_nodes = []
        self.err = ErrorHandling()
        for i in graph.inputs:
            edge_name = i[0]
            input_layer = cvt.make_input(i)
            layers.append(input_layer)
            exist_edges.append(i[0])
            graph.channel_dims[edge_name] = graph.shape_dict[edge_name][1]

        for id, node in enumerate(graph.nodes):
            node_name = node.name
            op_type = node.op_type
            inputs = node.inputs
            inputs_tensor = node.input_tensors
            input_non_exist_flag = False

            if node.op_type == "Constant":
                continue
            if node.op_type == "Upsample":
                node.attrs["height_scale"] = 2
                node.attrs["width_scale"] = 2

            for inp in inputs:
                if inp not in exist_edges and inp not in inputs_tensor:
                    input_non_exist_flag = True
                    break
            if input_non_exist_flag:
                continue

            if op_type not in cvt._ONNX_NODE_REGISTRY:
                self.err.unsupported_op(node)
                continue
            converter_fn = cvt._ONNX_NODE_REGISTRY[op_type]
            layer = converter_fn(node, graph, self.err)
            if type(layer) == tuple:
                for l in layer:
                    layers.append(l)
            else:
                layers.append(layer)
            outs = node.outputs
            for out in outs:
                exist_edges.append(out)
        return layers


def main():
    print("process start...")
    options = parse_arguments()
    converter = OnnxConvertCaffe(options.input_path)
    if options.proto_path is None:
        converter.convert_caffe()
    elif os.path.exists(options.proto_path):
        converter.convert_weights(options.proto_path)
    else:
        print("input error")
    print("process end!")


if __name__ == "__main__":
    main()
