from xflowrl.graphs.nasbench import NasbenchGraph
from nasbench import api



def main():
    ng = NasbenchGraph()

    model_spec = api.ModelSpec(
        # Adjacency matrix of the module
        matrix=[[0, 1, 1, 1, 0, 1, 0],  # input layer
                [0, 0, 0, 0, 0, 0, 1],  # 1x1 conv
                [0, 0, 0, 0, 0, 0, 1],  # 3x3 conv
                [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 0, 1],  # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 0, 1],  # 3x3 max-pool
                [0, 0, 0, 0, 0, 0, 0]],  # output layer
        # Operations at the vertices of the module, matches order of matrix
        ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

    graph = ng.build_graph(model_spec)


if __name__ == '__main__':
    main()
