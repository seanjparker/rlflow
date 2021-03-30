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
from xflowrl.agents.network.mdrnn import gmm_loss


@tf.function
def update_step(agent, states, next_states, actions, terminals, rewards):
    with tf.GradientTape() as tape:
        latent_state = [agent.main_net.get_embeddings(x["graph"]) for x in states]
        next_latent_state = [agent.main_net.get_embeddings(x["graph"]) for x in next_states]
        mus, sigmas, log_pi, rs, ds = agent.mdrnn(actions, latent_state)
        gmm = gmm_loss(next_latent_state, mus, sigmas, log_pi)
        bce_f = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        bce = bce_f(terminals, ds)

        mse = tf.keras.losses.mse(rewards, rs)
        scale = agent.latent_size + 2
        loss = (gmm + bce + mse) / scale
    grads = tape.gradient(loss, agent.trunk.trainable_variables)
    agent.trunk_optimizer.apply_gradients(zip(grads, agent.trunk.trainable_variables))
    return loss


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

    states = []
    next_states = []
    rewards = []
    terminals = []
    actions = []

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
            xfer_action, loc_action = agent.act(states=state, explore=True)

            # Action delivered in shape (1,), need ()
            next_state, reward, terminal, _ = env.step((xfer_action, loc_action))

            # Append to buffer.
            states.append(state)
            next_states.append(next_state)
            rewards.append(reward)
            terminals.append(terminal)

            state = next_state
            timestep += 1

            # If terminal, reset.
            if terminal:
                timestep = 0

                # print(f'Final runtime:\t{final_runtime:.4f}')
                # print(f'Difference:\t'
                #      f'{final_runtime - start_runtime:+.4f} ({(final_runtime - start_runtime) / start_runtime:+.2%})')
                # print(xfers_applied)
                # print('-' * 40)

                # Do an update after collecting specified number of batches.
                # This is a hyper-parameter that will require a lot of experimentation.
                # One episode could be one rewrite of the graph, and it may be desirable to perform a small
                # update after every rewrite.
                if current_episode > 0 and current_episode % episodes_per_batch == 0:
                    # print('Finished episode = {}, Mean reward for last {} episodes = {}'.format(
                    #    current_episode, episodes_per_batch, np.mean(episode_rewards[-episodes_per_batch:])))

                    loss = update_step(agent, states, next_states, actions, terminals, rewards)
                    print(f'loss = {loss}')

                    # Reset buffers.
                    states = []
                    next_states = []
                    rewards = []
                    terminals = []
                    actions = []

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
