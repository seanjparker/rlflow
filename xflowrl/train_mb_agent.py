import argparse
import copy
import json
import sys
from datetime import datetime
import os
import tensorflow as tf
import numpy as np

from xflowrl.agents.mb_agent import MBAgent
from xflowrl.environment.worldmodel import WorldModelEnvironment
from xflowrl.graphs.util import load_graph


def main(_args):
    graph_name, graph = load_graph(_args.graph)

    path_prefix = f'logs/xflowrl_mb_ctrl/{graph_name}/'
    if _args.timestamp:
        path_prefix += _args.timestamp
        timestamp = _args.timestamp
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

    num_locations = 200

    env = WorldModelEnvironment(num_locations=num_locations)
    env.set_graph(graph)
    init_graph = env.reset()  # Need to do this to get the number of actions

    num_actions = env.get_num_actions()

    num_episodes = 2000  # Todo: change
    episodes_per_batch = 10  # Todo: change

    hparams = dict(
        batch_size=episodes_per_batch,
        num_actions=num_actions,
        num_locations=num_locations,
        reducer=tf.math.unsorted_segment_sum,
        # Typically use small learning rates, depending on problem try [0.0025 - 0.00001]
        pi_learning_rate=1e-3,
        vf_learning_rate=1e-3,
        message_passing_steps=5,
        network_name=graph_name,
        checkpoint_timestamp=timestamp,
        wm_timestamp=_args.wm_timestamp
    )

    agent = MBAgent(**hparams)
    agent.load_wm()
    start_episode = int(agent.ckpt.step)
    print(f'Starting from episode = {start_episode}')

    # Init the world model environment after we have created the agent
    env.init_state(agent.main_net, agent.mdrnn)

    with open(info_filename, 'wt') as fp:
        hp = copy.deepcopy(hparams)
        hp['reducer'] = 'tf.unsorted_segment_sum'
        json.dump({
            'hparams': hp,
            'graphs': [graph_name]
        }, fp)

    print(f'Output filename: {output_filename}')
    output_file = open(output_filename, 'at')

    states = []
    xfer_actions = []
    xfer_log_probs = []
    xfer_vf_values = []

    loc_actions = []
    loc_log_probs = []
    loc_vf_values = []

    rewards = []
    terminals = []
    episode_rewards = []

    print(f'Training on graph: {graph_name}')
    for current_episode in range(start_episode, num_episodes):
        terminal = False
        episode_reward = 0
        timestep = 0

        # Returns the current state in latent space -- tensor shape (1, latent_size)
        state = env.reset_wm(init_graph)

        while not terminal:
            xfer_action, xfer_log_prob, xfer_vf_value, \
                loc_action, loc_log_prob, loc_vf_value = agent.act(state, explore=True)

            # Action delivered in shape (1,), need ()
            next_state, reward, terminal, _ = env.step((xfer_action, loc_action))

            # Append to buffer.
            states.append(state)
            xfer_actions.append(xfer_action)
            xfer_log_probs.append(xfer_log_prob)
            xfer_vf_values.append(xfer_vf_value)
            loc_actions.append(loc_action)
            loc_log_probs.append(loc_log_prob)
            loc_vf_values.append(loc_vf_value)

            rewards.append(reward)
            terminals.append(terminal)

            state = next_state
            episode_reward += reward
            timestep += 1

            # If terminal, reset.
            if terminal:
                timestep = 0
                episode_rewards.append(episode_reward)

                if current_episode > 0 and current_episode % episodes_per_batch == 0:
                    print('Finished episode = {}, Mean reward for last {} episodes = {}'.format(
                        current_episode, episodes_per_batch, np.mean(episode_rewards[-episodes_per_batch:])))

                    xfer_policy_loss, xfer_vf_loss, loc_policy_loss, loc_vf_loss, info = agent.update(
                        states, xfer_actions, xfer_log_probs, xfer_vf_values,
                        loc_actions, loc_log_probs, loc_vf_values, rewards, terminals)
                    print(f'policy loss = {xfer_policy_loss}, vf loss = {xfer_vf_loss}')
                    print(f'sub policy loss = {loc_policy_loss}, sub vf loss = {loc_vf_loss}')

                    # Reset buffers
                    states = []
                    xfer_actions = []
                    xfer_log_probs = []
                    xfer_vf_values = []
                    loc_actions = []
                    loc_log_probs = []
                    loc_vf_values = []
                    rewards = []
                    terminals = []

                    # Log to tensorboard
                    with train_summary_writer.as_default():
                        tf.summary.scalar('episode_reward', episode_reward, step=current_episode)
                        tf.summary.scalar('policy_loss', xfer_policy_loss, step=current_episode)
                        tf.summary.scalar('vf_loss', xfer_vf_loss, step=current_episode)
                        tf.summary.scalar('sub_policy_loss', loc_policy_loss, step=current_episode)
                        tf.summary.scalar('sub_vf_loss', loc_vf_loss, step=current_episode)
                        for k, v in info.items():
                            tf.summary.scalar(k, v, step=current_episode)

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
    parser.add_argument('--wm_timestamp',
                        help='Timestamp of the checkpoint that contains the MDRNN model in the format YYYYMMDD-HHMMSS')
    args = parser.parse_args(sys.argv[1:])
    main(args)
