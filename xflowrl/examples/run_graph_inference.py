import numpy as np
import tensorflow as tf

from xflowrl.agents.hierarchical_agent import HierarchicalAgent
from xflowrl.environment.hierarchical import HierarchicalEnvironment
from xflowrl.graphs.util import load_graph
import taso as ts
import sys
import argparse
import json
import os


# Load graph
#   - Graph name
#   - Timestamp to evaluate
# Perform inference on graph and record avg over 100 runs w/o optimisations
# Run TASO on graph and record results
# Load checkpoint of optimized graph
# Call run_memorysafe 100 times and average results
# Note: After each step, write the results to a file (in case of crash)


def save_record(results, graph_name, timestamp):
    print(results)
    print('Writing results to file...')
    path = f'./results/{graph_name}/{timestamp}/results.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w+') as f:
        json.dump(results, f)
    print('Done!')


def store_runtimes(name, runtimes):
    return {name: dict(
        zip(('mean', 'std', 'min', 'max'),
            (str(np.mean(runtimes)), str(np.std(runtimes)), str(np.min(runtimes)), str(np.max(runtimes)))))}


def get_xflowrl_runtime(graph, graph_name, checkpoint):
    env = HierarchicalEnvironment(real_measurements=True)
    env.set_graph(graph)
    env.reset()

    num_actions = env.get_num_actions()

    hparams = dict(
        num_actions=num_actions,
        num_locations=100,
        discount=0.99,
        gae_lambda=0.97,
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
    agent.load()

    env.set_graph(graph)
    state = env.reset()

    start_runtime = env.get_cost()
    t_reward = 0
    terminal = False
    while not terminal:
        main_action, _, _, sub_action, _, _ = agent.act(states=state, explore=False)
        next_state, rew, terminal, _ = env.step((main_action, sub_action))
        state = next_state
        t_reward += rew

    final_runtime = env.get_cost()
    print(f'Estimated runtime: {(final_runtime - start_runtime) / start_runtime}')
    runtimes = np.zeros(100, dtype=np.float32)
    for i in range(100):
        runtimes[i] = env.get_cost(real_measurement=True)
    return runtimes


def main(graph_name_or_path, timestamp):
    graph_name, graph = load_graph(graph_name_or_path)
    results = {graph_name: []}

    # Get the base runtime of the graph w/o optimisations applied
    runtimes = np.zeros(100, dtype=np.float32)
    print(f'Getting runtime on {graph_name} for baseline')
    for i in range(100):
        runtimes[i] = graph.run_time_memorysafe()
    results[graph_name].append(store_runtimes('baseline', runtimes))
    save_record(results, graph_name, timestamp)

    # Get the optimised runtime using TASO
    ts_optimized = ts.optimize(graph)
    runtimes = np.zeros(100, dtype=np.float32)
    print(f'Getting runtime on {graph_name} for TASO')
    for i in range(100):
        runtimes[i] = ts_optimized.run_time_memorysafe()
    results[graph_name].append(store_runtimes('taso', runtimes))
    save_record(results, graph_name, timestamp)

    # Get the optimised runtime using xflowrl (loading from checkpoint)
    print(f'Getting runtime on {graph_name} for xflowrl')
    graph_name, graph = load_graph(graph_name_or_path)
    runtimes = get_xflowrl_runtime(graph, graph_name, timestamp)
    # graph_name, graph = load_graph(f'./models/{graph_name}/{timestamp}/{graph_name}.onnx')
    # for i in range(100):
    #    runtimes[i] = graph.run_time_memorysafe()
    results[graph_name].append(store_runtimes('xflowrl', runtimes))
    save_record(results, graph_name, timestamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True, help='Name of the graph, or file path')
    parser.add_argument('--timestamp', required=True,
                        help='Timestamp of the checkpoint to evaluate in the format YYYYMMDD-HHMMSS')
    args = parser.parse_args(sys.argv[1:])
    main(args.graph, args.timestamp)
