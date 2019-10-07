import sys
import time
import onnx


import taso as ts


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

    #output_filename = 'results_{:%Y-%m-%d_%H-%M-%S}.csv'.format(datetime.now())
    #print("Output filename: {}".format(output_filename))
    #output_file = open(output_filename, 'wt')

    for current_graph_file, current_graph in graphs:

        print("Training on graph: {}".format(current_graph_file))

        start_runtime = current_graph.cost()

        start_time = time.time()
        optimized_graph = ts.optimize(current_graph, alpha=1.02, budget=100_000)
        time_taken_taso = time.time() - start_time

        final_runtime_taso = optimized_graph.cost()

        print("-"*40)
        print("Optimized graph {} in ".format(current_graph_file))
        print("Time taken for TASO search: {:.2f} seconds".format(time_taken_taso))
        print("Observed speedup for TASO search: {:.4f} ms (final runtime: {:.4f}).".format(
            start_runtime - final_runtime_taso, final_runtime_taso))

        print("Measured runtime: {:.4f} ms".format(optimized_graph.run_time()))

        #optimized_onnx = ts.export_onnx(optimized_graph)
        #onnx.save(optimized_onnx, "./taso_optimized_{}".format(current_graph_file))

    # output_file.close()
    # Export trained model to current directory with checkpoint name "mymodel".
    # agent.save("mymodel")


if __name__ == '__main__':
    main(sys.argv)
