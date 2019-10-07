import argparse

import sys
import time

import taso as ts

from nasbench import api

from xflowrl.analysis.complexity import graph_complexity
from xflowrl.analysis.taso import run_taso_optimize
from xflowrl.analysis.xflowrl import run_xflowrl_optimize
from xflowrl.environment.hierarchical import HierarchicalEnvironment
from xflowrl.graphs.nasbench import NasbenchTASO


INPUT = 'input'
OUTPUT = 'output'
IDENTITY = 'identity'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
BOTTLENECK3X3 = 'bottleneck3x3'
BOTTLENECK5x5 = 'bottleneck5x5'
MAXPOOL3x3CONV = 'maxpool3x3-conv1x1'

NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cont', default=None, help='Date to continue from')
    args = parser.parse_args(argv[1:])

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

    env = HierarchicalEnvironment()

    for current_graph_file, current_graph in graphs:
        graph_complexity(current_graph, current_graph_file, env=env)

    run_xflowrl_optimize(graphs, env=env, cont=args.cont)


if __name__ == '__main__':
    main(sys.argv)
