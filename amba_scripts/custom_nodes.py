#!/usr/bin/python3

import sys, os, subprocess

lib_path = subprocess.check_output(['tv2', '-basepath', 'AmbaCnnUtils'])
sys.path.append(lib_path.decode().rstrip('\n') + '/parser/caffe/')

import caffe_common as cfc

node_list = []

def parse_permute_node(layer_obj):

    node = {}
    src_list = []
    permute_attr = [''] * 5

    # initialize node - id, type, description
    node = cfc.init_node(layer_obj.name, 'Permute', layer_obj.name)

    #Find the correct parent node for a given node
    src_list.append(cfc.get_parent_node(layer_obj.bottom[0]))
    node['src'] = src_list.copy()

    #Update blob-layer dictionary for your custom node
    cfc.add_to_blob_layer_dict(layer_obj.top[0], layer_obj.name)

    if len(layer_obj.permute_param.order):
        permute_attr[0] = layer_obj.permute_param.order[0]
        permute_attr[1] = layer_obj.permute_param.order[1]
        permute_attr[2] = layer_obj.permute_param.order[2]
        permute_attr[3] = layer_obj.permute_param.order[3]

    node['attr_str'] = " ".join(str(x) for x in permute_attr)

    cfc.node_list.append(node)

    return node['id']


def parse_reorg_node(layer_obj):

    node = {}
    src_list = []
    reorg_attr = [''] * 1

    # initialize node - id, type, description
    node = cfc.init_node(layer_obj.name, 'Reorg', layer_obj.name)

    #Find the correct parent node for a given node
    src_list.append(cfc.get_parent_node(layer_obj.bottom[0]))
    node['src'] = src_list.copy()

    #Update blob-layer dictionary for your custom node
    cfc.add_to_blob_layer_dict(layer_obj.top[0], layer_obj.name)
    reorg_attr[0] = layer_obj.reorg_param.stride

    node['attr_str'] = " ".join(str(x) for x in reorg_attr)
    cfc.node_list.append(node)
    return node['id']

# Entry function for custom node parser
# User is expected to implement "parse_custom_node" function
def parse_custom_node(layer, caffe_net=None, coeff_folder=None):
    node_id = None
    layer_type = layer.type

    if layer_type == 'Permute':
        node_id = parser_permute_node(layer)
    elif layer_type == 'Reorg':
        node_id = parse_reorg_node(layer)
    else:
        node_id = 'CUSTOM_OP_NO_MATCH'

    return node_id
