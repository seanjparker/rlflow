import tensorflow as tf

from tensorflow.python.framework import ops as tf_ops

from nasbench.lib import base_ops
from nasbench.lib.model_builder import build_module

from tf2onnx import optimizer, utils
from tf2onnx.tfonnx import process_tf_graph, tf_optimize
from tf2onnx.loader import freeze_session

from xflowrl.graphs.onnx import load_onnx_model


class NasbenchTASO(object):
    def __init__(self):
        pass

    def build_model(self, spec, options=None):
        tf_graph, inputs, outputs = self.build_graph(spec, options=options)

        print(inputs.name)
        print(outputs.name)

        input_names = ['input:0']  # [128, 32, 32, 3]
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
        # utils.save_protobuf('/tmp/tmponnx.onnx', model_proto)

        return load_onnx_model(model_proto)

    def build_graph(self, spec, is_training=False, options=None):
        g = tf_ops.Graph()
        with g.as_default():

            config = {
                'num_stacks': 3,
                'num_modules_per_stack': 3,
                'stem_filter_size': 128,
                'data_format': 'channels_last',
                'num_labels': 5000
            }
            if config['data_format'] == 'channels_last':
                channel_axis = 3
            elif config['data_format'] == 'channels_first':
                channel_axis = 1
            else:
                raise ValueError('invalid data_format')

            if options:
                config.update(options)

            inputs = tf.constant(30, shape=(128, 32, 32, 3), dtype=tf.dtypes.float32, name='input')

            # Initial stem convolution
            with tf.variable_scope('stem'):
                net = base_ops.conv_bn_relu(
                    inputs, 3, config['stem_filter_size'],
                    is_training, config['data_format'])

            for stack_num in range(config['num_stacks']):
                channels = net.get_shape()[channel_axis].value

                # Downsample at start (except first)
                if stack_num > 0:
                    net = tf.layers.max_pooling2d(
                        inputs=net,
                        pool_size=(2, 2),
                        strides=(2, 2),
                        padding='same',
                        data_format=config['data_format'])

                    # Double output channels each time we downsample
                    channels *= 2

                with tf.variable_scope('stack{}'.format(stack_num)):
                    for module_num in range(config['num_modules_per_stack']):
                        with tf.variable_scope('module{}'.format(module_num)):
                            net = build_module(
                                spec,
                                inputs=net,
                                channels=channels,
                                is_training=is_training)

            # Global average pool
            if config['data_format'] == 'channels_last':
                net = tf.reduce_mean(net, [1, 2])
            elif config['data_format'] == 'channels_first':
                net = tf.reduce_mean(net, [2, 3])
            else:
                raise ValueError('invalid data_format')

            # Fully-connected layer to labels
            logits = tf.layers.dense(
                inputs=net,
                units=config['num_labels'],
                name='output'
            )

        return g, inputs, logits


