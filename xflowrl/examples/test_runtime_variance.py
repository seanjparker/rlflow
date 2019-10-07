import sys
import time
import onnx


import taso as ts
import numpy as np

from xflowrl.environment.hierarchical import HierarchicalEnvironment


def load_graph(filename):
    return ts.load_onnx(filename)


def main(argv):
    #graph_files = glob.glob('graphs/**/*.onnx', recursive=True)
    graph_files = ['graphs/squeezenet1.1.onnx']
    #graph_files = ['taso_optimized_graphs/squeezenet1.1.onnx']
    #graph_files = ['/tmp/tmponnx.onnx']

    skip_files = []#{'graphs/vgg16.onnx', 'graphs/inception_v2.onnx', 'graphs/resnet34v1.onnx'}

    graphs = []
    for graph_file in graph_files:
        if graph_file in skip_files:
            continue
        print("Loading graph: {}".format(graph_file))
        graphs.append((graph_file, load_graph(graph_file)))
        if graph_file == 'graphs/squeezenet1.1.onnx':
            break

    #env = HierarchicalEnvironment(real_measurements=True)

    for current_graph_file, current_graph in graphs:
        #env.set_graph(current_graph)
        # env.reset()

        runtimes = []

        for i in range(100):
            if i % 10 == 0:
                print("Measurement {}".format(i))
            runtimes.append(current_graph.run_time_memorysafe())

        print("Runtimes: {:.4f} ({:.4f})".format(np.mean(runtimes), np.std(runtimes)))


if __name__ == '__main__':
    main(sys.argv)
