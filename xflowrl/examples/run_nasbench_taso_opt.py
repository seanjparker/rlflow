import sys
import time

import taso as ts
import onnx

from nasbench import api

from xflowrl.analysis.complexity import graph_complexity
from xflowrl.analysis.taso import run_taso_optimize
from xflowrl.graphs.nasbench import NasbenchTASO


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


def main(argv):
    graphs = []

    model_spec = api.ModelSpec(
        # Adjacency matrix of the module
        matrix=[[0, 1, 1, 1, 1, 1, 0],  # input layer
                [0, 0, 1, 0, 0, 0, 1],  # 1x1 conv
                [0, 0, 0, 1, 0, 0, 1],  # 3x3 conv
                [0, 0, 0, 0, 1, 0, 1],  # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 1, 1],  # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 0, 1],  # 3x3 max-pool
                [0, 0, 0, 0, 0, 0, 0]],  # output layer
        # Operations at the vertices of the module, matches order of matrix
        ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

    ng = NasbenchTASO()
    graph = ng.build_model(model_spec, {
        'num_stacks': 3,
        'num_modules_per_stack': 3,
    })
    graphs.append(('NASbench-3x3', graph))

    for current_graph_file, current_graph in graphs:
        graph_complexity(current_graph, current_graph_file)
        optimized_graph, _, _, _ = run_taso_optimize(current_graph, current_graph_file, budget=150)

        optimized_onnx = ts.export_onnx(optimized_graph)
        onnx.save(optimized_onnx, "./taso_optimized_{}.onnx".format(current_graph_file))


if __name__ == '__main__':
    main(sys.argv)
