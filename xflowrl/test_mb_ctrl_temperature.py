import argparse
import copy
from datetime import datetime
import json
import os
import sys
import tensorflow as tf
import numpy as np

from xflowrl.agents.mb_agent import MBAgent
from xflowrl.environment.worldmodel import WorldModelEnvironment
from xflowrl.graphs.util import load_graph


def main(_args):
    temperature = float(_args.temperature)
    graph_name, graph = load_graph(_args.graph)

    path_prefix = f'logs/xflowrl_mb_ctrl_temp/{graph_name}/'
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
        wm_timestamp=_args.wm_timestamp,
        temperature=temperature
    )

    agent = MBAgent(**hparams)
    agent.load_wm()
    start_episode = int(agent.ckpt.step)
    print(f'Starting from episode = {start_episode}')

    # Init the world model environment after we have created the agent
    env.init_state(agent.main_net, agent.mdrnn)

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

    best_real_improv = 9999.99
    best_pred_improv = 9999.99

    print(f'Training on graph: {graph_name}')
    for current_episode in range(start_episode, num_episodes):
        terminal = False
        episode_reward = 0
        timestep = 0

        # Returns the current state in latent space -- tensor shape (1, latent_size)
        state = env.reset_wm(init_graph)

        # Reset real env, used for getting action masks
        env.set_graph(graph)
        real_state = env.reset()
        start_real_runtime = env.get_cost()
        masks = dict(xfer_mask=real_state['mask'], loc_mask=real_state['location_mask'])

        while not terminal:
            xfer_action, xfer_log_prob, xfer_vf_value, \
                loc_action, loc_log_prob, loc_vf_value = agent.act(dict(state=state, masks=masks), explore=True)

            # Action delivered in shape (1,), need ()
            next_state, reward, terminal, next_masks = env.step((xfer_action, loc_action))

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
            masks = next_masks
            last_real_reward = next_masks['real_reward']
            episode_reward += reward
            timestep += 1

            # If terminal, reset.
            if terminal:
                print(f'Episode: {current_episode}, timesteps: {timestep}')
                timestep = 0
                episode_rewards.append(episode_reward)

                pred_runtime_diff = rewards[-1] - rewards[0]
                pred_percent_improvement = pred_runtime_diff / rewards[0]

                real_runtime_diff = last_real_reward - start_real_runtime
                real_percent_improvement = real_runtime_diff / start_real_runtime

                best_pred_improv = min(best_pred_improv, pred_percent_improvement)
                best_real_improv = min(best_real_improv, real_percent_improvement)

                print(f'Predicted Runtime Improvement:\t'
                      f'{pred_runtime_diff:+.4f} ({pred_percent_improvement:+.2%})')
                print(f'Real Runtime Improvement:\t'
                      f'{real_runtime_diff:+.4f} ({real_percent_improvement:+.2%})')
                print('-' * 40)

                if current_episode > 0 and current_episode % episodes_per_batch == 0:
                    print('Finished episode = {}, Mean reward for last {} episodes = {}'.format(
                        current_episode, episodes_per_batch, np.mean(episode_rewards[-episodes_per_batch:])))

                    xfer_policy_loss, xfer_vf_loss, loc_policy_loss, loc_vf_loss, _ = agent.update(
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
                        # tf.summary.scalar('policy_loss', xfer_policy_loss, step=current_episode)
                        # tf.summary.scalar('vf_loss', xfer_vf_loss, step=current_episode)
                        # tf.summary.scalar('sub_policy_loss', loc_policy_loss, step=current_episode)
                        # tf.summary.scalar('sub_vf_loss', loc_vf_loss, step=current_episode)
                        # for k, v in info.items():
                        #    tf.summary.scalar(k, v, step=current_episode)

                    agent.save()
                    print(f'Checkpoint Episode = {int(agent.ckpt.step)}')
                    print(f'Best pred improvement: {best_pred_improv:+.2%}')
                    print(f'Best real improvement: {best_real_improv:+.2%}')
        agent.ckpt.step.assign_add(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True, help='Name of the graph, or file path')
    parser.add_argument('--timestamp',
                        help='Timestamp of the checkpoint to evaluate in the format YYYYMMDD-HHMMSS')
    parser.add_argument('--wm_timestamp',
                        help='Timestamp of the checkpoint that contains the MDRNN model in the format YYYYMMDD-HHMMSS')
    parser.add_argument('--temperature',
                        help='Temperature to use when training the controller inside the world model')
    args = parser.parse_args(sys.argv[1:])
    # for t in [0.1, 0.5, 0.75, 1.0, 1.2, 1.5, 1.75, 2, 2.5, 3]:
    main(args)
