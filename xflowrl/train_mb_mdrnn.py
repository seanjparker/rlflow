import argparse
import copy
import json
import sys
from datetime import datetime
import os
import tensorflow as tf

from xflowrl.agents.mb_agent import MBAgent
from xflowrl.environment.hierarchical import HierarchicalEnvironment
from xflowrl.graphs.util import load_graph


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

    num_locations = 200

    env = HierarchicalEnvironment(num_locations=num_locations, real_measurements=False, reward_function=None)
    env.set_graph(graph)
    env.reset()  # Need to do this to get the number of actions

    num_actions = env.get_num_actions()

    num_episodes = 2000  # Todo: change
    episodes_per_batch = 10  # Todo: change

    hparams = dict(
        batch_size=episodes_per_batch,
        num_actions=num_actions,
        num_locations=num_locations,
        reducer=tf.math.unsorted_segment_sum,
        # Typically use small learning rates, depending on problem try [0.0025 - 0.00001]
        gmm_learning_rate=0.0025,
        message_passing_steps=5,
        network_name=graph_name,
        checkpoint_timestamp=timestamp
    )

    agent = MBAgent(**hparams)
    agent.load()
    start_episode = int(agent.ckpt.step)
    print(f'Starting from episode = {start_episode}')

    states = []
    states_batch = []
    next_states = []
    next_states_batch = []
    rewards = []
    terminals = []
    xfer_actions = []
    loc_actions = []
    xfer_action_batch = []
    loc_action_batch = []

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

        env.set_graph(graph)

        state = env.reset()
        start_runtime = env.get_cost()
        print(f'Start runtime: {start_runtime:.4f}')

        timestep = 0
        while not terminal:
            # Random action agent
            xfer_action, loc_action = agent.act(state, explore=True)

            # Action delivered in shape (1,), need ()
            next_state, reward, terminal, _ = env.step((xfer_action, loc_action))

            # Append to buffer.
            states.append(state)
            next_states.append(next_state)
            rewards.append(reward)
            terminals.append(terminal)
            xfer_actions.append(xfer_action[0])
            loc_actions.append(loc_action[0])

            state = next_state
            timestep += 1

            # If terminal, reset.
            if terminal:
                timestep = 0

                final_runtime = env.get_cost()
                print(f'Final runtime:\t{final_runtime:.4f}')
                print(f'Difference:\t'
                      f'{final_runtime - start_runtime:+.4f} ({(final_runtime - start_runtime) / start_runtime:+.2%})')
                print('-' * 40)
                # l1 = max(len(a) for a in actions)
                # filled_actions = [a + [151] * (l1 - len(a)) for a in actions]
                # filled_actions = actions + [151] * (20 - len(actions))
                xfer_action_batch.append(xfer_actions.copy())
                xfer_actions = []
                loc_action_batch.append(loc_actions.copy())
                loc_actions = []
                states_batch.append(states.copy())
                states = []
                next_states_batch.append(next_states.copy())
                next_states = []

                if current_episode > 0 and current_episode % episodes_per_batch == 0:
                    # Calculate loss for mini-batch rollout using the random agent
                    loss = agent.update_step(states_batch, next_states_batch,
                                             xfer_action_batch, loc_action_batch, terminals, rewards)
                    print(f'MDRNN Loss = {loss}')

                    # Reset buffers.
                    states_batch = []
                    next_states_batch = []
                    rewards = []
                    terminals = []
                    xfer_action_batch = []
                    loc_action_batch = []

                    detailed_costs.append(env.get_detailed_costs())
                    with open(runtime_info_filename, 'w', encoding='utf-8') as f:
                        json.dump(detailed_costs, f, ensure_ascii=False, indent=4)

                    # Log to tensorboard
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=current_episode)

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
