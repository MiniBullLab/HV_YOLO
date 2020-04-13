#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from optparse import OptionParser
import pathlib
import tensorflow as tf
import tf2onnx
from tf2onnx import loader, optimizer
from tf2onnx.tfonnx import tf_optimize


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is tensorflow convert to onnx"

    parser.add_option("-w", "--pb", dest="pb_path",
                      metavar="PATH", type="string", default=None,
                      help="tensorflow pb path")

    (options, args) = parser.parse_args()

    if options.pb_path:
        if not os.path.exists(options.pb_path):
            parser.error("Could not find the onnx file")
        else:
            options.pb_path = os.path.normpath(options.pb_path)
    else:
        parser.error("'pb_path' option is required to run this program")

    return options


class TensorflowConvertOnnx():

    def __init__(self, pb_path):
        self.target_opset = 10
        self.pb_path = pathlib.Path(pb_path)
        self.input_names = ["net_input:0"]
        self.output_names = ["conv2d_10/Sigmoid:0"]
        self.onnx_save_path = self.pb_path.with_suffix(".onnx")

    def convert_onnx(self):
        graph_def, inputs, outputs = loader.from_graphdef(str(self.pb_path),
                                                          self.input_names,
                                                          self.output_names)
        graph_def = tf_optimize(inputs, outputs, graph_def, True)
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name='')
        with tf.Session(graph=tf_graph):
            temp_graph = tf2onnx.tfonnx.process_tf_graph(tf_graph,
                                                         input_names=self.input_names,
                                                         output_names=self.output_names,
                                                         opset=self.target_opset)
        onnx_graph = optimizer.optimize_graph(temp_graph)
        model_proto = onnx_graph.make_model("converted from {}".format(str(self.pb_path)))
        tf2onnx.utils.save_protobuf(str(self.onnx_save_path), model_proto)

    def print_op(self):
        sess = tf.Session()

        output_graph_def = tf.GraphDef()
        with open(str(self.pb_path), "rb") as f:
            output_graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(output_graph_def, name="")
        op = sess.graph.get_operations()
        for m in op:
            print(m.values())
        sess.close()


def main():
    print("process start...")
    options = parse_arguments()
    converter = TensorflowConvertOnnx(options.pb_path)
    converter.convert_onnx()
    print("process end!")


if __name__ == "__main__":
    main()
