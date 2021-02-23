import numpy as np
import tensorflow as tf

from xflowrl.agents.hierarchical_agent import HierarchicalAgent
from xflowrl.environment.hierarchical import HierarchicalEnvironment
from xflowrl.graphs.util import load_graph
import taso as ts
import sys
import argparse
import json
import time


# Load graph
#   - Graph name
#   - Timestamp to evaluate
# Perform inference on graph and record avg over 100 runs w/o optimisations
# Run TASO on graph and record results
# Load checkpoint of optimized graph
# Call run_memorysafe 100 times and average results
# Note: After each step, write the results to a file (in case of crash)


def save_record(results, graph_name, timestamp):
    with open(f'results/{graph_name}/{timestamp}/results.json', 'w') as outfile:
        json.dump(results, outfile)


def store_runtimes(name, runtimes):
    return {name: dict(
        zip(('mean', 'std', 'min', 'max'), (np.mean(runtimes), np.std(runtimes), np.min(runtimes), np.max(runtimes))))}


def get_xflowrl_runtime(graph, graph_name, checkpoint):
    env = HierarchicalEnvironment(real_measurements=True)
    env.set_graph(graph)
    state = env.reset()

    num_actions = env.get_num_actions()

    hparams = dict(
        num_actions=num_actions,
        num_locations=100,
        discount=0.99,
        gae_lambda=1.0,
        reducer=tf.math.unsorted_segment_sum,
        # Typically use small learning rates, depending on problem try [0.0025 - 0.00001]
        learning_rate=0.0025,
        # Value function can have the same or a slightly more aggressive learning rate.
        vf_learning_rate=0.0025,
        policy_layer_size=32,
        # This limits the aggressiveness of the update -> 0.2 is often the default value, 0.3
        # for a more aggressive update, 0.1 for a more conservative one.
        clip_ratio=0.3,
        message_passing_steps=5,
        network_name=graph_name,
        checkpoint_timestamp=checkpoint
    )

    agent = HierarchicalAgent(**hparams)
    agent.act(state, explore=True)
    agent.load()

    env.set_graph(graph)
    state = env.reset()

    terminal = False
    while not terminal:
        main_action, _, _, sub_action, _, _ = agent.act(states=state, explore=False)
        next_state, _, terminal, _ = env.step((main_action, sub_action))
        state = next_state

    runtimes = np.zeros(100, dtype=np.float32)
    for i in range(100):
        runtimes[i] = env.get_cost(real_measurement=True)
    return runtimes


def main(graph_name_or_path, timestamp):
    graph_name, graph = load_graph(graph_name_or_path)
    results = dict(graph_name=dict())

    # Get the base runtime of the graph w/o optimisations applied
    runtimes = np.zeros(100, dtype=np.float32)
    for i in range(100):
        runtimes[i] = graph.run_time_memorysafe()
    results[graph_name] = store_runtimes('baseline', runtimes)
    save_record(results, graph_name, timestamp)

    # Get the optimised runtime using TASO
    ts_optimized = ts.optimize(graph)
    runtimes = np.zeros(100, dtype=np.float32)
    for i in range(100):
        runtimes[i] = ts_optimized.run_time_memorysafe()
    results[graph_name] = store_runtimes('taso', runtimes)
    save_record(results, graph_name, timestamp)

    # Get the optimised runtime using xflowrl (loading from checkpoint)
    runtimes = get_xflowrl_runtime(graph, graph_name, timestamp)
    for i in range(100):
        runtimes[i] = ts_optimized.run_time_memorysafe()
    results[graph_name] = store_runtimes('xflowrl', runtimes)
    save_record(results, graph_name, timestamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True, help='Name of the graph, or file path')
    parser.add_argument('--timestamp', required=True,
                        help='Timestamp of the checkpoint to evaluate in the format YYYYMMDD-HHMMSS')
    args = parser.parse_args(sys.argv[1:])
    main(args.graph, args.timestamp)
