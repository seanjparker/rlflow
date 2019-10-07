import copy
import json
import sys
import glob
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from xflowrl.agents.hierarchical_agent import HierarchicalAgent
from xflowrl.analysis.complexity import graph_complexity
from xflowrl.environment.hierarchical import HierarchicalEnvironment

import taso as ts

from xflowrl.graphs.bert import build_graph_bert


def load_graph(filename):
    return ts.load_onnx(filename)


def main(argv):
    #graph = build_graph_bert()
    #graph_files = glob.glob('graphs/**/*.onnx', recursive=True)
    graph_files = ['graphs/squeezenet1.1.onnx', 'graphs/resnet18v1.onnx', 'graphs/resnet34v1.onnx', 'graphs/resnet50v1.onnx', 'graphs/resnet152v1.onnx']
    #graph_files = ['graphs/resnet50v1.onnx']
    skip_files = []#{'graphs/vgg16.onnx', 'graphs/inception_v2.onnx', 'graphs/resnet34v1.onnx'}

    graphs = []
    for graph_file in graph_files:
        if graph_file in skip_files:
            continue
        print("Loading graph: {}".format(graph_file))
        graphs.append((graph_file, load_graph(graph_file)))

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
