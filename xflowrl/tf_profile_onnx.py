import argparse
import os
import sys

import tensorflow as tf
import onnx
import taso as ts
from onnx_tf.backend import prepare

from tensorflow.python.profiler import profiler_v2 as profiler

from xflowrl.graphs.util import load_graph

INFERENCE_STEPS = 1000


def convert_onnx_to_tf(onnx_path, pb_path):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model, auto_cast=True)
    tf_rep.export_graph(pb_path)


def main(_args):
    if _args.graph:
        graph_name, graph = load_graph(_args.graph)
        onnx_model = ts.export_onnx(graph)
        path = f'./graphs/{graph_name}_compiled.onnx'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        onnx.save(onnx_model, path)
        return

    shape = _args.shape[1:-1].replace(" ", "").split(",")  # convert format [dim_0, dim_1, dim_2] into array
    shape = [int(v) for v in shape]

    print("="*40)
    print("Converting ONNX model to TensorFlow SavedModel")
    print("="*40)

    convert_onnx_to_tf(_args.onnx, _args.pb)
    model = tf.saved_model.load(_args.pb)
    f = model.signatures["serving_default"]
    features = tf.random.uniform(shape, dtype=tf.float32, name="data")

    print("="*40)
    print("Running profiler on TensorFlow model")
    print("="*40)

    profiler.warmup()
    profiler.start(_args.logdir)
    for step in range(INFERENCE_STEPS):
        with tf.profiler.experimental.Trace("train", step_num=step):
            _ = f(data=features)
    profiler.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=False, help='Optional argument to convert Python model to ONNX')
    require_others = '--graph' not in sys.argv
    parser.add_argument('--onnx', required=require_others, help='File path to onnx model')
    parser.add_argument('--pb', required=require_others, help='File path for protobuf model')
    parser.add_argument('--logdir', required=require_others, help='Path to log profiler results in tensorboard')
    parser.add_argument('--shape', required=require_others, help='Shape of input in format [batch_size, dim_0, ...]')

    args = parser.parse_args(sys.argv[1:])
    main(args)


# python xflowrl/tf_profile_onnx.py --onnx "./graphs/squeezenet1.1.onnx" --pb "./graphs/compiled/squeezenet1.1.onnx" --logdir "./logs/profile" --shape "[1, 3, 224, 224]"
# python xflowrl/tf_profile_onnx.py --onnx "./graphs/resnet18.onnx" --pb "./graphs/compiled/resnet18.onnx" --logdir "./logs/profile" --shape "[1, 3, 224, 224]"
# python xflowrl/tf_profile_onnx.py --onnx "./graphs/resnet50.onnx" --pb "./graphs/compiled/resnet50.onnx" --logdir "./logs/profile" --shape "[1, 3, 224, 224]"
# python xflowrl/tf_profile_onnx.py --onnx "./graphs/InceptionV3_compiled.onnx" --pb "./graphs/compiled/inceptionv3.onnx" --logdir "./logs/profile" --shape "[1, 3, 229, 229]"
# python xflowrl/tf_profile_onnx.py --onnx "./graphs/BERT_compiled.onnx" --pb "./graphs/compiled/bert.onnx" --logdir "./logs/profile" --shape "[64, 1024]"
