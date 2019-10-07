import copy
import json
import sys
import glob
import time
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf

from xflowrl.agents.hierarchical_agent import HierarchicalAgent
from xflowrl.environment.hierarchical import HierarchicalEnvironment
from xflowrl.graphs.bert import build_graph_bert

import taso as ts

from xflowrl.graphs.nasnet import build_graph_nasnet


def load_graph(filename):
    return ts.load_onnx(filename)


def main(argv):
    #graph = build_graph_bert()
    #graph_files = glob.glob('graphs/**/*.onnx', recursive=True)
    graph_files = ['graphs/squeezenet1.1.onnx']#, 'graphs/resnet18v1.onnx']
    #graph_files = ['/tmp/tmponnx.onnx']
    skip_files = []#{'graphs/vgg16.onnx', 'graphs/inception_v2.onnx', 'graphs/resnet34v1.onnx'}

    graphs = []
    for graph_file in graph_files:
        if graph_file in skip_files:
            continue
        print("Loading graph: {}".format(graph_file))
        graphs.append((graph_file, load_graph(graph_file)))

    #graph_files = []
    #graphs = []

    #graph_files.append('NASnet')
    #graphs.append(('NASnet', build_graph_nasnet()))

    # graph_files.append('BERT')
    # graphs.append(('BERT', build_graph_bert()))

    graph_file, graph = graphs[0]
    # graph = load_graph(graph_files[0])

    env = HierarchicalEnvironment(real_measurements=True)
    env.set_graph(graph)
    env.reset()  # Need to do this to get the number of actions1

    num_actions = env.get_num_actions()

    hparams = dict(
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
        clip_ratio=0.3,
        message_passing_steps=5
    )

    agent = HierarchicalAgent(**hparams)

    num_episodes = 2000  # Todo: change

    # How often will we update?
    episodes_per_batch = 10  # Todo: change

    # Demonstrating a simple training loop - collect a few samples, transform them to graph inputs,
    # update occasionally.
    episode_rewards = []

    # Storing samples.
    states = []
    main_actions = []
    main_log_probs = []
    main_baseline_values = []

    sub_actions = []
    sub_log_probs = []
    sub_baseline_values = []

    rewards = []
    terminals = []

    # Rewards should increase after a few hundred episodes.
    # This is not particularly tuned, and clearly the way the state is converted to a graph
    # is rather questionable, but it demonstrates all relevant mechanisms in principle.
    now = datetime.now()

    output_filename = 'results_{:%Y-%m-%d_%H-%M-%S}.csv'.format(now)
    info_filename = 'info_{:%Y-%m-%d_%H-%M-%S}.txt'.format(now)
    save_model_filename = 'saved_model_{:%Y-%m-%d_%H-%M-%S}'.format(now)

    logger_inference = logging.getLogger('log_inference')
    logger_inference.addHandler(logging.FileHandler('log_training_{:%Y-%m-%d_%H-%M-%S}'.format(now)))
    logger_inference.setLevel(logging.INFO)

    with open(info_filename, 'wt') as fp:
        hp = copy.deepcopy(hparams)
        hp['reducer'] = 'tf.unsorted_segment_sum'
        json.dump({
            'hparams': hp,
            'graphs': graph_files
        }, fp)

    print("Output filename: {}".format(output_filename))
    output_file = open(output_filename, 'wt')

    for current_episode in range(num_episodes):
        # Keep stepping
        terminal = False
        episode_reward = 0

        # Re-assign state to initial state.
        #current_graph_file = graph_files[current_episode % len(graph_files)]
        #current_graph = ts.load_onnx(current_graph_file)

        current_graph_file, current_graph = graphs[current_episode % len(graphs)]
        print("Training on graph: {}".format(current_graph_file))
        env.set_graph(current_graph)

        #ts.optimize(current_graph)
        #print("Optimized")
        #continue

        state = env.reset()
        start_runtime = env.get_cost()
        print("Start runtime: {:.4f}".format(start_runtime))

        timestep = 0
        while not terminal:
            main_action, main_log_prob, main_baseline_value, sub_action, sub_log_prob, sub_baseline_value = agent.act(states=state, explore=True)

            # Action delivered in shape (1,), need ()
            next_state, reward, terminal, _ = env.step((main_action, sub_action))

            # Append to buffer.
            states.append(state)

            # Main action
            main_actions.append(main_action)
            main_log_probs.append(main_log_prob)
            main_baseline_values.append(main_baseline_value)

            # Sub action
            sub_actions.append(sub_action)
            sub_log_probs.append(sub_log_prob)
            sub_baseline_values.append(sub_baseline_value)

            rewards.append(reward)
            terminals.append(terminal)

            state = next_state
            episode_reward += reward

            logger_inference.info("Episode {}. Iteration {}. Graph: {}. XFER: {}. Location: {}. Reward: {:.6f}. Terminal: {}".format(
                current_episode,
                timestep,
                current_graph_file,
                main_action,
                sub_action,
                reward,
                terminal
            ))
            timestep += 1

            # If terminal, reset.
            if terminal:
                timestep = 0
                episode_rewards.append(episode_reward)

                output_file.write('{:.2f},{:.4f},{:.4f}\n'.format(time.time(), episode_reward, env.get_cost()))
                output_file.flush()

                # Env reset is handled in outer loop

                final_runtime = env.get_cost()

                print("Final runtime: {:.4f}".format(final_runtime))
                print("Difference:   {:+.4f}".format(final_runtime - start_runtime))
                print("-"*40)

                # Do an update after collecting specified number of batches.
                # This is a hyper-parameter that will require a lot of experimentation.
                # One episode could be one rewrite of the graph, and it may be desirable to perform a small
                # update after every rewrite.
                if current_episode > 0 and current_episode % episodes_per_batch == 0:
                    # CartPole runs very short episodes so we only report average across last batch.
                    print('Finished episode = {}, Mean reward for last {} episodes = {}'.format(
                        current_episode, episodes_per_batch, np.mean(episode_rewards[-episodes_per_batch:])))
                    # Simply pass collected trajectories to the agent for a single update.
                    loss, baseline_loss, sub_loss, sub_baseline_loss = agent.update(
                        states=states,
                        actions=main_actions,
                        log_probs=main_log_probs,
                        baseline_values=main_baseline_values,
                        sub_actions=sub_actions,
                        sub_log_probs=sub_log_probs,
                        sub_baseline_values=sub_baseline_values,
                        rewards=rewards,
                        terminals=terminals
                    )
                    # Loss should be decreasing.
                    print("Loss = {}, baseline loss = {}".format(loss, baseline_loss))
                    print("Sub loss = {}, Sub baseline loss = {}".format(sub_loss, sub_baseline_loss))
                    # Reset buffers.
                    states = []
                    main_actions = []
                    main_log_probs = []
                    main_baseline_values = []
                    sub_actions = []
                    sub_log_probs = []
                    sub_baseline_values = []
                    rewards = []
                    terminals = []

                    agent.save('./model/{}'.format(save_model_filename))

    output_file.close()
    agent.save('./model/{}'.format(save_model_filename))
    # Export trained model to current directory with checkpoint name "mymodel".
    # agent.save("mymodel")


if __name__ == '__main__':
    main(sys.argv)
