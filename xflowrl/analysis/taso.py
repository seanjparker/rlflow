import time

import taso as ts


def run_taso_optimize(graph, name='Untitled', alpha=1.0, budget=1000):
    print("Training on graph: {}".format(name))

    start_runtime = graph.cost()

    start_time = time.time()
    optimized_graph = ts.optimize(graph, alpha=alpha, budget=budget)
    time_taken_taso = time.time() - start_time

    final_runtime_taso = optimized_graph.cost()

    print("-" * 40)
    print("Optimized graph {} in ".format(name))
    print("Time taken for TASO search: {:.2f} seconds".format(time_taken_taso))
    print("Observed speedup for TASO search: {:.4f} seconds (final runtime: {:.4f}).".format(
        start_runtime - final_runtime_taso, final_runtime_taso))

    return optimized_graph, start_runtime, final_runtime_taso, time_taken_taso
