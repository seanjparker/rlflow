import argparse
import copy
import json
import sys
import time
from datetime import datetime
import os

import numpy as np
import tensorflow as tf

from xflowrl.agents.mb_agent import MBAgent
from xflowrl.environment.hierarchical import HierarchicalEnvironment
from xflowrl.graphs.util import load_graph
from xflowrl.util.util import plot_xfer_heatmap, plot_to_image


def main(path_or_name, cont=None):
    graph_name, graph = load_graph(path_or_name)

    path_prefix = f'logs/xflowrl_mb/{graph_name}/'
    if cont:
        path_prefix += cont
        timestamp = cont
        print('Continuing provided log')
    else:
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        path_prefix += current_time
        timestamp = current_time
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        print(f'Created Tensorboard log directory: {path_prefix}')

    output_filename = f'{path_prefix}/results.csv'
    info_filename = f'{path_prefix}/info.txt'
    runtime_info_filename = f'{path_prefix}/runtime_info.json'
    train_log_dir = f'{path_prefix}/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # A custom reward function can be provided to the environment to replace the default
    # incremental reward function
    def custom_reward(last_runtime, norm_costs):
        new_runtime, flops, mem_acc, num_kernels = norm_costs['runtime'], norm_costs['flops'], \
                                                   norm_costs['mem_acc'], norm_costs['num_kernels']
        return new_runtime * 0.5 + mem_acc * 0.5

    num_locations = 200

    env = HierarchicalEnvironment(num_locations=num_locations, real_measurements=False, reward_function=None)
    env.set_graph(graph)
    env.reset()  # Need to do this to get the number of actions

    num_actions = env.get_num_actions()

    hparams = dict(
        num_actions=num_actions,
        num_locations=num_locations,
        reducer=tf.math.unsorted_segment_sum,
        # Typically use small learning rates, depending on problem try [0.0025 - 0.00001]
        learning_rate=0.0025,
        message_passing_steps=5,
        network_name=graph_name,
        checkpoint_timestamp=timestamp
    )

    num_episodes = 2000  # Todo: change

    # How often will we update?
    episodes_per_batch = 10  # Todo: change

    agent = MBAgent(**hparams)
    agent.load()
    start_episode = int(agent.ckpt.step)
    print(f'Starting from episode = {start_episode}')

    # Demonstrating a simple training loop - collect a few samples, transform them to graph inputs,
    # update occasionally.
    episode_rewards = []

    # Storing samples.
    states = []
    xfer_actions = []
    loc_actions = []

    rewards = []
    terminals = []

    xfers_applied = {}
    detailed_costs = []

    with open(info_filename, 'wt') as fp:
        hp = copy.deepcopy(hparams)
        hp['reducer'] = 'tf.unsorted_segment_sum'
        json.dump({
            'hparams': hp,
            'graphs': [graph_name]
        }, fp)

    print(f'Output filename: {output_filename}')
    output_file = open(output_filename, 'at')

    try:
        with open(runtime_info_filename, 'r', encoding='utf-8') as f:
            detailed_costs = json.load(f)
    except FileNotFoundError:
        detailed_costs = []

    print(f'Training on graph: {graph_name}')
    for current_episode in range(start_episode, num_episodes):
        # Keep stepping
        terminal = False
        episode_reward = 0
        xfers_applied = {}

        env.set_graph(graph)

        state = env.reset()
        start_runtime = env.get_cost()
        print(f'Start runtime: {start_runtime:.4f}')

        timestep = 0
        while not terminal:
            xfer_action, loc_action = agent.act(states=state, explore=True)

            # Action delivered in shape (1,), need ()
            next_state, reward, terminal, _ = env.step((xfer_action, loc_action))

            # Append to buffer.
            states.append(state)

            # Main action
            xfer_actions.append(xfer_action)
            # Sub action
            loc_actions.append(loc_action)

            rewards.append(reward)
            terminals.append(terminal)

            state = next_state
            episode_reward += reward

            # Store the xfer applied
            if str(xfer_action) not in xfers_applied:
                xfers_applied[str(xfer_action)] = 0
            xfers_applied[str(xfer_action)] += 1

            timestep += 1

            # If terminal, reset.
            if terminal:
                timestep = 0
                episode_rewards.append(episode_reward)

                output_file.write('{:.2f},{:.4f},{:.4f}\n'.format(time.time(), episode_reward, env.get_cost()))
                output_file.flush()

                # Env reset is handled in outer loop
                final_runtime = env.get_cost()

                print(f'Final runtime:\t{final_runtime:.4f}')
                print(f'Difference:\t'
                      f'{final_runtime - start_runtime:+.4f} ({(final_runtime - start_runtime) / start_runtime:+.2%})')
                print(xfers_applied)
                print('-' * 40)

                # Do an update after collecting specified number of batches.
                # This is a hyper-parameter that will require a lot of experimentation.
                # One episode could be one rewrite of the graph, and it may be desirable to perform a small
                # update after every rewrite.
                if current_episode > 0 and current_episode % episodes_per_batch == 0:
                    # CartPole runs very short episodes so we only report average across last batch.
                    print('Finished episode = {}, Mean reward for last {} episodes = {}'.format(
                        current_episode, episodes_per_batch, np.mean(episode_rewards[-episodes_per_batch:])))
                    # Simply pass collected trajectories to the agent for a single update.
                    gmm_loss, xfer_c_loss, loc_c_loss, info = agent.update(
                        states=states,
                        actions=xfer_actions,
                        rewards=rewards,
                        terminals=terminals
                    )
                    # Loss should be decreasing.
                    print(f'gmm loss = {gmm_loss}')
                    print(f'xfer controller loss = {xfer_c_loss}, loc controller loss = {loc_c_loss}')

                    # Reset buffers.
                    states = []
                    xfer_actions = []
                    loc_actions = []
                    rewards = []
                    terminals = []

                    detailed_costs.append(env.get_detailed_costs())
                    with open(runtime_info_filename, 'w', encoding='utf-8') as f:
                        json.dump(detailed_costs, f, ensure_ascii=False, indent=4)

                    # Log to tensorboard
                    with train_summary_writer.as_default():
                        tf.summary.scalar('episode_reward', episode_reward, step=current_episode)
                        tf.summary.scalar('gmm_loss', gmm_loss, step=current_episode)
                        tf.summary.scalar('xfer_controller_loss', xfer_c_loss, step=current_episode)
                        tf.summary.scalar('loc_controller_loss', loc_c_loss, step=current_episode)
                        for k, v in info.items():
                            tf.summary.scalar(k, v, step=current_episode)

                        figure = plot_xfer_heatmap(xfers_applied)
                        tf.summary.image('xfers Applied', plot_to_image(figure), max_outputs=10, step=current_episode)

                    agent.save()
                    print(f'Checkpoint Episode = {int(agent.ckpt.step)}')
        agent.ckpt.step.assign_add(1)

    output_file.close()
    agent.export(env.graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True, help='Name of the graph, or file path')
    parser.add_argument('--timestamp',
                        help='Timestamp of the checkpoint to evaluate in the format YYYYMMDD-HHMMSS')
    args = parser.parse_args(sys.argv[1:])
    main(args.graph, args.timestamp)
