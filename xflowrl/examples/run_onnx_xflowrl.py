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
from xflowrl.graphs.util import load_graph_by_name, load_graph_from_file


def main(name, path, cont=None):
    graph_name = graph = None
    if name is not None:
        graph_name, graph = load_graph_by_name(name)
    elif path is not None:
        graph_name, graph = load_graph_from_file(path)

    # Rewards should increase after a few hundred episodes.
    # This is not particularly tuned, and clearly the way the state is converted to a graph
    # is rather questionable, but it demonstrates all relevant mechanisms in principle.
    logger_inference = logging.getLogger('log_inference')
    logger_inference.setLevel(logging.INFO)

    timestamp = None
    if cont:
        output_filename = f'results_{cont}.csv'
        info_filename = f'info_{cont}.txt'
        logger_inference.addHandler(logging.FileHandler(f'log_training_{cont}'))
        if not os.path.isfile(output_filename):
            print(f'Continue error: No such file {output_filename}')
            return None, None, None

        train_log_dir = f'logs/xflowrl/{graph_name}/{cont}/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        timestamp = cont
        print("Continuing provided log")
    else:
        now = datetime.now()

        output_filename = f'results_{now:%Y%m%d-%H%M%S}.csv'
        info_filename = f'info_{now:%Y%m%d-%H%M%S}.txt'
        logger_inference.addHandler(logging.FileHandler(f'log_training_{now:%Y%m%d-%H%M%S}'))

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = f'logs/xflowrl/{graph_name}/{current_time}/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        timestamp = current_time
        print(f'Created Tensorboard log directory: {train_log_dir}')

    # A custom reward function can be provided to the environment to replace the default
    # incremental reward function
    def custom_reward(last_runtime, new_runtime):
        return last_runtime - new_runtime

    num_locations = 200

    env = HierarchicalEnvironment(num_locations=num_locations, real_measurements=False, reward_function=None)
    env.set_graph(graph)
    env.reset()  # Need to do this to get the number of actions

    num_actions = env.get_num_actions()

    hparams = dict(
        num_actions=num_actions,
        num_locations=num_locations,
        discount=0.99,
        gae_lambda=1.0,
        reducer=tf.math.unsorted_segment_sum,
        # Typically use small learning rates, depending on problem try [0.0025 - 0.00001]
        learning_rate=0.0025,
        # Value function can have the same or a slightly more aggressive learning rate.
        vf_learning_rate=0.01,
        policy_layer_size=32,
        # This limits the aggressiveness of the update -> 0.2 is often the default value, 0.3
        # for a more aggressive update, 0.1 for a more conservative one.
        clip_ratio=0.3,
        message_passing_steps=5
    )

    num_episodes = 2000  # Todo: change

    # How often will we update?
    episodes_per_batch = 10  # Todo: change

    agent = HierarchicalAgent(**hparams)
    agent.load()
    start_episode = int(agent.ckpt.step)
    print("Starting from episode = {}".format(start_episode))

    # Demonstrating a simple training loop - collect a few samples, transform them to graph inputs,
    # update occasionally.
    episode_rewards = []

    # Storing samples.
    states = []
    main_actions = []
    main_log_probs = []
    main_vf_values = []

    sub_actions = []
    sub_log_probs = []
    sub_vf_values = []

    rewards = []
    terminals = []

    with open(info_filename, 'wt') as fp:
        hp = copy.deepcopy(hparams)
        hp['reducer'] = 'tf.unsorted_segment_sum'
        json.dump({
            'hparams': hp,
            'graphs': [graph_name]
        }, fp)

    print(f'Output filename: {output_filename}')
    output_file = open(output_filename, 'wt')

    print(f'Training on graph: {graph_name}')

    for current_episode in range(start_episode, num_episodes):
        # Keep stepping
        terminal = False
        episode_reward = 0

        env.set_graph(graph)

        state = env.reset()
        start_runtime = env.get_cost()
        print("Start runtime: {:.4f}".format(start_runtime))

        timestep = 0
        while not terminal:
            main_action, main_log_prob, main_vf_value, \
            sub_action, sub_log_prob, sub_vf_value = agent.act(states=state, explore=True)

            # Action delivered in shape (1,), need ()
            next_state, reward, terminal, _ = env.step((main_action, sub_action))

            # Append to buffer.
            states.append(state)

            # Main action
            main_actions.append(main_action)
            main_log_probs.append(main_log_prob)
            main_vf_values.append(main_vf_value)

            # Sub action
            sub_actions.append(sub_action)
            sub_log_probs.append(sub_log_prob)
            sub_vf_values.append(sub_vf_value)

            rewards.append(reward)
            terminals.append(terminal)

            state = next_state
            episode_reward += reward

            logger_inference.info(
                "Episode {}. Iteration {}. Graph: {}. XFER: {}. Location: {}. Reward: {:.6f}. Terminal: {}".format(
                    current_episode,
                    timestep,
                    graph_name,
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
                print("Difference:   {:+.4f} ({:+.2%})".format(
                    final_runtime - start_runtime, ((final_runtime - start_runtime) / start_runtime)))
                print("-" * 40)

                # Do an update after collecting specified number of batches.
                # This is a hyper-parameter that will require a lot of experimentation.
                # One episode could be one rewrite of the graph, and it may be desirable to perform a small
                # update after every rewrite.
                if current_episode > 0 and current_episode % episodes_per_batch == 0:
                    # CartPole runs very short episodes so we only report average across last batch.
                    print('Finished episode = {}, Mean reward for last {} episodes = {}'.format(
                        current_episode, episodes_per_batch, np.mean(episode_rewards[-episodes_per_batch:])))
                    # Simply pass collected trajectories to the agent for a single update.
                    policy_loss, vf_loss, sub_policy_loss, sub_vf_loss, info = agent.update(
                        states=states,
                        actions=main_actions,
                        log_probs=main_log_probs,
                        vf_values=main_vf_values,
                        sub_actions=sub_actions,
                        sub_log_probs=sub_log_probs,
                        sub_vf_values=sub_vf_values,
                        rewards=rewards,
                        terminals=terminals
                    )
                    # Loss should be decreasing.
                    print("policy loss = {}, vf loss = {}".format(policy_loss, vf_loss))
                    print("sub policy loss = {}, sub vf loss = {}".format(sub_policy_loss, sub_vf_loss))

                    # Reset buffers.
                    states = []
                    main_actions = []
                    main_log_probs = []
                    main_vf_values = []
                    sub_actions = []
                    sub_log_probs = []
                    sub_vf_values = []
                    rewards = []
                    terminals = []

                    # Log to tensorboard
                    with train_summary_writer.as_default():
                        tf.summary.scalar('episode_reward', episode_reward, step=current_episode)
                        tf.summary.scalar('policy_loss', policy_loss, step=current_episode)
                        tf.summary.scalar('vf_loss', vf_loss, step=current_episode)
                        tf.summary.scalar('sub_policy_loss', sub_policy_loss, step=current_episode)
                        tf.summary.scalar('sub_vf_loss', sub_vf_loss, step=current_episode)
                        for k, v in info.items():
                            tf.summary.scalar(k, v, step=current_episode)

                    agent.save()
                    print("Checkpoint Episode = {}".format(int(agent.ckpt.step)))

        agent.ckpt.step.assign_add(1)

    output_file.close()
    agent.save()


if __name__ == '__main__':
    main(sys.argv)
