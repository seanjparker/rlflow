"""
OpenAI gym cartpole - balancing a cartpole. The state has 4 dimensions and is normally
an array. There are only two actions to control the pole (left/right):

https://github.com/openai/gym/wiki/CartPole-v0

To test the graph agent, the state is mapped to 4 nodes instead.

Note for context:

CartPole itself is not a graph problem, this is purely meant as a sanity check that the
update mechanism works as intended. Hence, if the agent is not learning on another problem,
this is then likely a configuration/feature/insufficient training samples issue, not
a logical problem in the agent.
"""
import sys
import gym
import numpy as np
import graph_nets as gn
import tensorflow as tf
from xflowrl.agents.agent import Agent


def cartpole_state_to_graph(state):
    """
    Maps the state array to a graph tuple. Since there is nothing to be
    semantically encoded into a graph, this is a dummy demo to show how to generate
    graph inputs.

    Computationally, we would typically create a matrix for each of the graph tuple elements, then
    update them in place as we modify the graph.

    Observe that this implementation requires static graph layouts, i.e. if the graph has a variable number of nodes
    and edges, create one large matrix and and zero out unused nodes/graphs by:
        - Updating the senders/receivers list to rewrite their edges.
        - For example, if a graph has 95 nodes, but the matrix was constructed with 100 nodes, nodes 96-100
        - can simply set all their features to zero, and any corresponding edges can be rewired to go from
        - e.g. node 96 to node 96 (self-loop) with edge features 0.

    This way, unused 'slots' in the node matrix do not contribute to the aggregations.

    Args:
        state (ndarray): State vector with 4 elements.

    Returns:
        gn.GraphTuple.
    """
    # Add state to globals as well.
    globals = np.asarray([state], dtype=np.float32)

    # State has 4 elements, simply wrote each to a node.
    nodes = np.asarray([
        # Note extra [] - batch dim required for node/edge features.
        [state[0]],
        [state[1]],
        [state[2]],
        [state[3]],
    ], dtype=np.float32)
    # These are just two example edges without purpose.
    # There is nothing to encode on edges here.
    edges = np.asarray(
        [
            [0.0, 0.0],
        ],
        dtype=np.float32
    )
    receivers = [0]
    senders = [1]
    n_node = [4]
    n_edge = [1]
    return {
        "graph": gn.graphs.GraphsTuple(nodes=nodes,
                                       edges=edges,
                                       globals=globals,
                                       receivers=receivers,
                                       senders=senders,
                                       n_node=n_node,
                                       n_edge=n_edge),
        # We never want to mask out an action here, both actions
        # always valid.
        "mask": [1, 1]
    }


def main(argv):
    env = gym.make("CartPole-v0")
    num_actions = env.action_space.n
    agent = Agent(
        num_actions=num_actions,
        discount=0.95,
        gae_lambda=0.97,
        reducer=tf.math.unsorted_segment_sum,
        # Typically use small learning rates, depending on problem try [0.0025 - 0.00001]
        # I did not tune them particularly for this toy problem.
        learning_rate=0.0025,
        # Value function can have the same or a slightly more aggressive learning rate.
        vf_learning_rate=0.03,
        policy_layer_size=32,
        # This limits the aggressiveness of the update -> 0.2 is often the default value, 0.3
        # for a more aggressive update, 0.1 for a more conservative one.
        clip_ratio=0.2
    )
    num_episodes = 1000

    # How often will we update?
    episodes_per_batch = 10

    # Demonstrating a simple training loop - collect a few samples, transform them to graph inputs,
    # update occasionally.
    episode_rewards = []

    # Storing samples.
    states = []
    actions = []
    log_probs = []
    vf_values = []
    rewards = []
    terminals = []

    # TODO: The commented out code below shows how to load from a checkpoint in eager mode.
    # TODO Test after exporting the model first.

    # We act once and discard output, so all eager variables are created.
    # There is nothing to load into otherwise.
    # Create a dummy input, call act.
    # graph_state = cartpole_state_to_graph(env.reset())
    # agent.act(states=graph_state, explore=True)
    # Import from checkpoint name.
    # agent.load("mymodel")

    # Rewards should increase after a few hundred episodes.
    # This is not particularly tuned, and clearly the way the state is converted to a graph
    # is rather questionable, but it demonstrates all relevant mechanisms in principle.
    for current_episode in range(num_episodes):
        # Keep stepping
        terminal = False
        episode_reward = 0
        state = env.reset()

        while not terminal:
            # Convert to graph.
            graph_state = cartpole_state_to_graph(state)

            # Important: When evaluating the final trained model, set explore=False for deterministic
            # actions.
            action, log_prob, value = agent.act(states=graph_state, explore=True)

            # Action delivered in shape (1,), need ()
            next_state, reward, terminal, _ = env.step(action[0])

            # Append to buffer.
            states.append(graph_state)
            actions.append(action)
            log_probs.append(log_prob)
            vf_values.append(value)
            rewards.append(reward)
            terminals.append(terminal)

            state = next_state
            episode_reward += reward

            # If terminal, reset.
            if terminal:

                episode_rewards.append(episode_reward)

                # Re-assign state to initial state.
                state = env.reset()

                # Do an update after collecting specified number of batches.
                # This is a hyper-parameter that will require a lot of experimentation.
                # One episode could be one rewrite of the graph, and it may be desirable to perform a small
                # update after every rewrite.
                if current_episode > 0 and current_episode % episodes_per_batch == 0:
                    # CartPole runs very short episodes so we only report average across last batch.
                    print('Finished episode = {}, Mean reward for last {} episodes = {}'.format(
                        current_episode, episodes_per_batch, np.mean(episode_rewards[-episodes_per_batch:])))
                    # Simply pass collected trajectories to the agent for a single update.
                    loss, vf_loss = agent.update(
                        states=states,
                        actions=actions,
                        log_probs=log_probs,
                        vf_values=vf_values,
                        rewards=rewards,
                        terminals=terminals
                    )
                    # Loss should be decreasing.
                    print("Loss = {}, vf loss = {}".format(loss, vf_loss))
                    # Reset buffers.
                    states = []
                    actions = []
                    log_probs = []
                    vf_values = []
                    rewards = []
                    terminals = []

    # Export trained model to current directory with checkpoint name "mymodel".
    agent.save("mymodel")


if __name__ == '__main__':
    main(sys.argv)
