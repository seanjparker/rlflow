import sys
import time
import onnx


import taso as ts


def load_graph(filename):
    return ts.load_onnx(filename)


def main(argv):
    graph_files = [
        # 'graphs/resnet50.onnx',
        # 'graphs/BERT_compiled.onnx',
        # 'graphs/InceptionV3_compiled.onnx',
        # 'graphs/resnet18.onnx',
        'graphs/squeezenet1.1.onnx'
    ]

    graphs = []
    for graph_file in graph_files:
        print("Loading graph: {}".format(graph_file))
        graphs.append((graph_file, load_graph(graph_file)))

    for current_graph_file, current_graph in graphs:

        print("Training on graph: {}".format(current_graph_file))

        start_runtime = current_graph.cost()

        start_time = time.time()
        optimized_graph = ts.optimize(current_graph, alpha=1.02, budget=200)
        time_taken_taso = time.time() - start_time

        final_runtime_taso = optimized_graph.cost()

        print("-"*40)
        print("Optimized graph {} in ".format(current_graph_file))
        print("Time taken for TASO search: {:.2f} seconds".format(time_taken_taso))
        print("Observed speedup for TASO search: {:.4f} ms (final runtime: {:.4f}).".format(
            start_runtime - final_runtime_taso, final_runtime_taso))

        print("Measured runtime: {:.4f} ms".format(optimized_graph.run_time()))


if __name__ == '__main__':
    main(sys.argv)
