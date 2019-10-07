import tensorflow as tf
from xflowrl.graphs.nasbench import NasbenchTASO
from nasbench import api

import onnx

from tf2onnx import optimizer, utils
from tf2onnx.tfonnx import process_tf_graph, tf_optimize
from tf2onnx.loader import freeze_session

from tensorflow.python.framework import ops as tf_ops


INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix


def main():
    ng = NasbenchTASO()

    model_spec = api.ModelSpec(
        # Adjacency matrix of the module
        matrix=[[0, 1, 1, 1, 0, 1, 0],  # input layer
                [0, 0, 0, 0, 0, 0, 1],  # 1x1 conv
                [0, 0, 0, 0, 0, 0, 1],  # 3x3 conv
                [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 0, 1],  # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 0, 1],  # 3x3 max-pool
                [0, 0, 0, 0, 0, 0, 0]],  # output layer
        # Operations at the vertices of the module, matches order of matrix
        ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

    tf_graph, inputs, outputs = ng.build_graph(model_spec)

    # tf.saved_model.save(graph, '/tmp/outmodel')

    print(inputs.name)
    print(outputs.name)

    input_names = ['input:0'] # [128, 32, 32, 3]
    output_names = ['output/BiasAdd:0']

    with tf.Session(graph=tf_graph) as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        frozen_graph_def = freeze_session(sess, output_names=['output/BiasAdd:0'])
        graph_def = tf_optimize(input_names, output_names, frozen_graph_def, True)

        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name='')

        g = process_tf_graph(tf_graph,
                             continue_on_error=True,
                             target=[],
                             opset=10,
                             custom_op_handlers={},
                             extra_opset=[],
                             shape_override=None,
                             input_names=input_names,
                             output_names=output_names,
                             inputs_as_nchw=None)

    onnx_graph = optimizer.optimize_graph(g)
    model_proto = onnx_graph.make_model('nasbench model')
    utils.save_protobuf('/tmp/tmponnx.onnx', model_proto)

    # onnx.save(onnx_graph, '/tmp/tmponnx')


if __name__ == '__main__':
    main()
