import sys
from xflowrl.analysis.complexity import graph_complexity
from xflowrl.environment.hierarchical import HierarchicalEnvironment

from xflowrl.graphs.util import load_graph


def main(argv):
    #graph_files = ['graphs/squeezenet1.1.onnx', 'graphs/resnet18v1.onnx', 'graphs/resnet34v1.onnx', 'graphs/resnet50v1.onnx', 'graphs/resnet152v1.onnx']
    #skip_files = []#{'graphs/vgg16.onnx', 'graphs/inception_v2.onnx', 'graphs/resnet34v1.onnx'}
    graph_files = ['InceptionV3']
    graphs = []
    for graph_file in graph_files:
        print("Loading graph: {}".format(graph_file))
        graphs.append((graph_file, load_graph(graph_file)[1]))

    #graph_files.append('BERT')
    #graphs.append(('BERT', build_graph_bert()))

    graph_file, graph = graphs[0]
    # graph = load_graph(graph_files[0])

    env = HierarchicalEnvironment()
    env.set_graph(graph)
    env.reset()  # Need to do this to get the number of actions1

    for current_graph_file, current_graph in graphs:
        graph_complexity(current_graph, current_graph_file, env)


if __name__ == '__main__':
    main(sys.argv)
