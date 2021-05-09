import argparse
import sys
import time

import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from statistics import mean

INFERENCE_STEPS = 100
WARMUP_STEPS = 20


def convert_onnx_to_tf(onnx_path, pb_path):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(pb_path)


def model_inf(infer_fn, shape):
    features = tf.random.uniform(shape)
    output_tensorname = list(infer_fn.structured_outputs.keys())[0]
    step_times = list()
    try:
        for step in range(1, INFERENCE_STEPS + 1):
            if step % 100 == 0:
                print("Processing step: %04d ..." % step)
            start_t = time.time()
            _ = infer_fn(features)[output_tensorname]
            step_time = time.time() - start_t
            if step >= WARMUP_STEPS:
                step_times.append(step_time)
    except tf.errors.OutOfRangeError:
        pass
    return step_times


def main(_args):
    onnx_path = _args.onnx
    pb_path = _args.pb
    trt_path = _args.trt
    shape = _args.shape[1:-1].replace(" ", "").split(",")  # convert format [dim_0, dim_1, dim_2] into array
    convert_onnx_to_tf(onnx_path, pb_path)

    params = trt.TrtConversionParams()
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=pb_path, conversion_params=params)
    converter.convert()
    converter.save(trt_path)

    model = tf.saved_model.load(trt_path)
    infer = model.signatures['serving_default']
    times = model_inf(infer, shape)
    avg_time = mean(times)
    print("\nAverage step time: %.1f msec" % (avg_time * 1e3))
    print("Average throughput: %d samples/sec" % (
           shape[0] / avg_time
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True, help='File path to onnx model')
    parser.add_argument('--pb', required=True, help='File path for protobuf model')
    parser.add_argument('--trt', required=True, help='File path for TensorRT model')
    parser.add_argument('--shape', required=True, help='Shape of input in format [batch_size, dim_0, ...]')

    args = parser.parse_args(sys.argv[1:])
    main(args)

