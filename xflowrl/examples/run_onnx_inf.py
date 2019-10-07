import sys
import glob
import time
from datetime import datetime
import logging
import onnx

import numpy as np
import tensorflow as tf

from xflowrl.agents.hierarchical_agent import HierarchicalAgent
from xflowrl.environment.hierarchical import HierarchicalEnvironment

import taso as ts

from xflowrl.graphs.bert import build_graph_bert


def load_graph(filename):
    return ts.load_onnx(filename)


def main(argv):
    graph_files = ['graphs/squeezenet1.1.onnx']
    skip_files = []#{'graphs/vgg16.onnx', 'graphs/inception_v2.onnx', 'graphs/resnet34v1.onnx'}

    model_file = './model/saved_model_2020-03-02_13-09-50'

    graphs = []
    #graph_files = []
    for graph_file in graph_files:
        if graph_file in skip_files:
            continue
        print("Loading graph: {}".format(graph_file))
        graphs.append((graph_file, load_graph(graph_file)))

    #graph_files = ['BERT']
    #graphs = [('BERT', build_graph_bert())]

    graph_file, graph = graphs[0]
    # graph = load_graph(graph_files[0])

    env = HierarchicalEnvironment()
    env.set_graph(graph)
    state = env.reset()  # Need to do this to get the number of actions1

    num_actions = env.get_num_actions()

    agent = HierarchicalAgent(
        num_actions=num_actions,
        num_locations=100,
        discount=0.99,
        gae_lambda=1.0,
        reducer=tf.unsorted_segment_sum,
        # Typically use small learning rates, depending on problem try [0.0025 - 0.00001]
        learning_rate=0.0025,
        # Value function can have the same or a slightly more aggressive learning rate.
        baseline_learning_rate=0.0025,
        policy_layer_size=32,
        # This limits the aggressiveness of the update -> 0.2 is often the default value, 0.3
        # for a more aggressive update, 0.1 for a more conservative one.
        clip_ratio=0.3
    )
    agent.act(state, explore=True)
    agent.load(model_file)

    logger_inference = logging.getLogger('log_inference')
    logger_inference.addHandler(logging.FileHandler('log_inference.txt'))
    logger_inference.setLevel(logging.INFO)

    episode_rewards = []
    for current_graph_file, current_graph in graphs:
        terminal = False
        episode_reward = 0

        print("Doing inference on graph: {}".format(current_graph_file))
        env.set_graph(current_graph)

        orig_onnx = ts.export_onnx(current_graph)
        onnx.save(orig_onnx, "./orig_{}".format(current_graph_file))

        state = env.reset()
        start_runtime = env.get_cost()
        print("Start runtime: {:.4f}".format(start_runtime))
        rewards = []

        start_time = time.time()

        while not terminal:
            main_action, main_log_prob, main_baseline_value, sub_action, sub_log_prob, sub_baseline_value = agent.act(states=state, explore=False)

            # Action delivered in shape (1,), need ()
            next_state, reward, terminal, _ = env.step((main_action, sub_action))

            rewards.append(reward)

            state = next_state
            episode_reward += reward

            logger_inference.info("Iteration {}. Graph: {}. XFER: {}. Location: {}. Reward: {:.6f}. Terminal: {}".format(
                len(rewards),
                current_graph_file,
                main_action,
                sub_action,
                reward,
                terminal
            ))

            # If terminal, reset.
            if terminal:
                episode_rewards.append(episode_reward)

                final_runtime = env.get_cost()

                print("Final runtime: {:.4f}".format(final_runtime))
                print("Difference:   {:+.4f}".format(final_runtime - start_runtime))
                print("-"*40)


        time_taken_rl = time.time() - start_time
        final_runtime_rl = env.get_cost()

        print("Time taken for RL inference: {:.2f} seconds".format(time_taken_rl))
        print("Observed speedup for RL inference: {:.4f} seconds (final runtime: {:.4f}).".format(
            final_runtime_rl - start_runtime, final_runtime_rl))
        final_graph_rl = env.graph

        optimized_onnx = ts.export_onnx(final_graph_rl)
        onnx.save(optimized_onnx, "./rl_optimized_{}".format(current_graph_file))

        # TASO
        start_time = time.time()
        optimized_graph = ts.optimize(current_graph, budget=1000)
        time_taken_taso = time.time() - start_time

        final_runtime_taso = optimized_graph.cost()

        optimized_onnx = ts.export_onnx(optimized_graph)
        onnx.save(optimized_onnx, "./taso_optimized_{}".format(current_graph_file))

        print("-"*40)
        print("Evaluated graph {}".format(current_graph_file))
        print("Start runtime: {:.4f}".format(start_runtime))
        print("Time taken for RL inference: {:.2f} seconds".format(time_taken_rl))
        print("Observed speedup for RL inference: {:.4f} seconds (final runtime: {:.4f}).".format(
            start_runtime - final_runtime_rl, final_runtime_rl))
        print("Real runtime of final RL graph: {:.4f}".format(final_graph_rl.run_time()))


        print("Time taken for TASO search: {:.2f} seconds".format(time_taken_taso))
        print("Observed speedup for TASO search: {:.4f} seconds (final runtime: {:.4f}).".format(
            start_runtime - final_runtime_taso, final_runtime_taso))
        start_measuretime = time.time()
        real_graph_runtime = optimized_graph.run_time()
        taken_measuretime = time.time() - start_measuretime
        print("Real runtime of final TASO graph: {:.4f} (measured in {:.4f}s)".format(real_graph_runtime, taken_measuretime))




if __name__ == '__main__':
    main(sys.argv)
