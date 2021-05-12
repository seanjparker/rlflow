import argparse
import sys
import time
import os
import tensorflow as tf
import numpy as np

from xflowrl.agents.mb_agent import MBAgent
from xflowrl.environment.worldmodel import WorldModelEnvironment
from xflowrl.graphs.util import load_graph


def main(_args):
    graph_name, graph = load_graph(_args.graph)

    path_prefix = f'results/test_reward/{graph_name}/'
    path_prefix += _args.wm_timestamp
    reward_filename = f'{path_prefix}/rewards.json'
    os.makedirs(os.path.dirname(reward_filename), exist_ok=True)

    num_locations = 200

    env = WorldModelEnvironment(num_locations=num_locations)
    env.set_graph(graph)
    init_graph = env.reset()  # Need to do this to get the number of actions

    num_actions = env.get_num_actions()

    num_episodes = 1  # Todo: change
    episodes_per_batch = 10  # Todo: change

    hparams = dict(
        batch_size=episodes_per_batch,
        num_actions=num_actions,
        num_locations=num_locations,
        reducer=tf.math.unsorted_segment_sum,
        message_passing_steps=5,
        network_name=graph_name,
        wm_timestamp=_args.wm_timestamp
    )

    agent = MBAgent(**hparams)
    agent.load_wm()
    start_episode = int(agent.ckpt.step)
    print(f'Starting from episode = {start_episode}')

    # Init the world model environment after we have created the agent
    env.init_state(agent.main_net, agent.mdrnn)

    print(f'Training on graph: {graph_name}')
    times = np.array([])
    start_time = time.time()
    for current_episode in range(start_episode, 11):
        terminal = False
        timestep = 0

        # Returns the current state in latent space -- tensor shape (1, latent_size)
        state = env.reset_wm(init_graph)

        # Reset real env, used for getting action masks
        env.set_graph(graph)
        real_state = env.reset()
        masks = dict(xfer_mask=real_state['mask'], loc_mask=real_state['location_mask'])

        while not terminal:
            xfer_action, _, _, \
                loc_action, _, _ = agent.act(dict(state=state, masks=masks), explore=True)

            # Action delivered in shape (1,), need ()
            env_step_start = time.time()
            next_state, reward, terminal, info = env.step((xfer_action, loc_action))
            diff = time.time() - env_step_start
            times = np.append(times, diff)
            print(f'Step: {diff}')

            state = next_state
            masks = dict(xfer_mask=info['xfer_mask'], loc_mask=info['loc_mask'])
            timestep += 1

            # If terminal, reset.
            if terminal:
                print(f'Episode: {current_episode}, timesteps: {timestep}')
                timestep = 0
    print(f'Average wall clock step: {times.mean()}')
    print(f'Wall clock total: {time.time() - start_time}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True, help='Name of the graph, or file path')
    parser.add_argument('--wm_timestamp',
                        help='Timestamp of the checkpoint that contains the MDRNN model in the format YYYYMMDD-HHMMSS')
    args = parser.parse_args(sys.argv[1:])
    main(args)
