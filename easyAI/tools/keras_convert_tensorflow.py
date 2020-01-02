#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from pathlib import Path
from absl import app
from absl import flags
from absl import logging
from keras import backend as K
from keras.models import model_from_json, model_from_yaml
from easyAI.converter.keras_models.model_factory import KerasModelFactory

K.set_learning_phase(0)
FLAGS = flags.FLAGS

flags.DEFINE_string('input_model', None, 'Path to the input model.')
flags.DEFINE_string('input_model_name', "FgSegNetV2", 'the input model name.')
flags.DEFINE_string('input_model_json', None, 'Path to the input model '
                                              'architecture in json format.')
flags.DEFINE_string('input_model_yaml', None, 'Path to the input model '
                                              'architecture in yaml format.')
flags.DEFINE_string('output_model', None, 'Path where the converted model will '
                                          'be stored.')
flags.DEFINE_boolean('save_graph_def', False,
                     'Whether to save the graphdef.pbtxt file which contains '
                     'the graph definition in ASCII format.')
flags.DEFINE_string('output_nodes_prefix', None,
                    'If set, the output nodes will be renamed to '
                    '`output_nodes_prefix`+i, where `i` will numerate the '
                    'number of of output nodes of the network.')
flags.DEFINE_boolean('quantize', False,
                     'If set, the resultant TensorFlow graph weights will be '
                     'converted from float into eight-bit equivalents. See '
                     'documentation here: '
                     'https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms')
flags.DEFINE_boolean('channels_first', False,
                     'Whether channels are the first dimension of a tensor. '
                     'The default is TensorFlow behaviour where channels are '
                     'the last dimension.')
flags.DEFINE_boolean('output_meta_ckpt', False,
                     'If set to True, exports the model as .meta, .index, and '
                     '.data files, with a checkpoint file. These can be later '
                     'loaded in TensorFlow to continue training.')

flags.mark_flag_as_required('input_model')


class KerasConvertTensorflow():

    channels_first = False
    input_model_json = None
    input_model_yaml = None
    output_nodes_prefix = None
    output_meta_ckpt = False
    save_graph_def = False
    quantize = False

    def __init__(self, h5_model_path, model_name):
        temp_output_path = Path(h5_model_path).with_suffix(".pb")
        self.output_model = str(temp_output_path)
        self.input_model = h5_model_path
        self.input_model_name = model_name

        self.keras_model_factory = KerasModelFactory()

    def keras_convert_tensorflow(self):
        # If output_model path is relative and in cwd, make it absolute from root
        output_model = self.output_model
        if str(Path(output_model).parent) == '.':
            output_model = str((Path.cwd() / output_model))

        output_fld = Path(output_model).parent
        output_model_name = Path(output_model).name
        output_model_stem = Path(output_model).stem
        output_model_pbtxt_name = output_model_stem + '.pbtxt'

        # Create output directory if it does not exist
        Path(output_model).parent.mkdir(parents=True, exist_ok=True)

        if KerasConvertTensorflow.channels_first:
            K.set_image_data_format('channels_first')
        else:
            K.set_image_data_format('channels_last')

        model = self.load_model(self.input_model, KerasConvertTensorflow.input_model_json,
                                KerasConvertTensorflow.input_model_yaml)

        # TODO(amirabdi): Support networks with multiple inputs
        orig_output_node_names = [node.op.name for node in model.outputs if node is not None]
        print(orig_output_node_names)
        if KerasConvertTensorflow.output_nodes_prefix:
            num_output = len(orig_output_node_names)
            pred = [None] * num_output
            converted_output_node_names = [None] * num_output

            # Create dummy tf nodes to rename output
            for i in range(num_output):
                converted_output_node_names[i] = '{}{}'.format(
                    KerasConvertTensorflow.output_nodes_prefix, i)
                pred[i] = tf.identity(model.outputs[i],
                                      name=converted_output_node_names[i])
        else:
            converted_output_node_names = orig_output_node_names
        logging.info('Converted output node names are: %s',
                     str(converted_output_node_names))

        sess = K.get_session()
        if KerasConvertTensorflow.output_meta_ckpt:
            saver = tf.train.Saver()
            saver.save(sess, str(output_fld / output_model_stem))

        if KerasConvertTensorflow.save_graph_def:
            tf.train.write_graph(sess.graph.as_graph_def(), str(output_fld),
                                 output_model_pbtxt_name, as_text=True)
            logging.info('Saved the graph definition in ascii format at %s',
                         str(Path(output_fld) / output_model_pbtxt_name))

        if KerasConvertTensorflow.quantize:
            from tensorflow.tools.graph_transforms import TransformGraph
            transforms = ["quantize_weights", "quantize_nodes"]
            transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [],
                                                   converted_output_node_names,
                                                   transforms)
            constant_graph = graph_util.convert_variables_to_constants(
                sess,
                transformed_graph_def,
                converted_output_node_names)
        else:
            constant_graph = graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(),
                converted_output_node_names)

        graph_io.write_graph(constant_graph, str(output_fld), output_model_name,
                             as_text=False)
        logging.info('Saved the freezed graph at %s',
                     str(Path(output_fld) / output_model_name))

    def load_model(self, input_model_path, input_json_path=None, input_yaml_path=None):
        if not Path(input_model_path).exists():
            raise FileNotFoundError(
                'Model file `{}` does not exist.'.format(input_model_path))
        try:
            model = self.keras_model_factory.load_model(input_model_path, self.input_model_name)
            return model
        except FileNotFoundError as err:
            logging.error('Input mode file (%s) does not exist.', self.input_model)
            raise err
        except ValueError as wrong_file_err:
            if input_json_path:
                if not Path(input_json_path).exists():
                    raise FileNotFoundError(
                        'Model description json file `{}` does not exist.'.format(
                            input_json_path))
                try:
                    model = model_from_json(open(str(input_json_path)).read())
                    model.load_weights(input_model_path)
                    return model
                except Exception as err:
                    logging.error("Couldn't load model from json.")
                    raise err
            elif input_yaml_path:
                if not Path(input_yaml_path).exists():
                    raise FileNotFoundError(
                        'Model description yaml file `{}` does not exist.'.format(
                            input_yaml_path))
                try:
                    model = model_from_yaml(open(str(input_yaml_path)).read())
                    model.load_weights(input_model_path)
                    return model
                except Exception as err:
                    logging.error("Couldn't load model from yaml.")
                    raise err
            else:
                logging.error(
                    'Input file specified only holds the weights, and not '
                    'the model definition. Save the model using '
                    'model.save(filename.h5) which will contain the network '
                    'architecture as well as its weights. '
                    'If the model is saved using the '
                    'model.save_weights(filename) function, either '
                    'input_model_json or input_model_yaml flags should be set to '
                    'to import the network architecture prior to loading the '
                    'weights. \n'
                    'Check the keras documentation for more details '
                    '(https://keras.io/getting-started/faq/)')
                raise wrong_file_err


def main(args):
    print("process start...")
    converter = KerasConvertTensorflow(FLAGS.input_model, FLAGS.input_model_name)
    KerasConvertTensorflow.channels_first = FLAGS.channels_first
    KerasConvertTensorflow.input_model_json = FLAGS.input_model_json
    KerasConvertTensorflow.input_model_yaml = FLAGS.input_model_yaml
    KerasConvertTensorflow.output_nodes_prefix = FLAGS.output_nodes_prefix
    KerasConvertTensorflow.output_meta_ckpt = FLAGS.output_meta_ckpt
    KerasConvertTensorflow.save_graph_def = FLAGS.save_graph_def
    KerasConvertTensorflow.quantize = FLAGS.quantize
    converter.keras_convert_tensorflow()
    print("process end!")


if __name__ == "__main__":
    app.run(main)