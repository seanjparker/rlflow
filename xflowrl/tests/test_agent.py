import unittest

from xflowrl.agents.agent import Agent
import graph_nets as gn
import numpy as np


class TestAgent(unittest.TestCase):
    """Tests basic functionality - do the API operations execute?"""
    NUM_ACTIONS = 5

    @staticmethod
    def random_state(num_actions):
        """
        Creates a random graph tuple.
        """
        globals = np.asarray([np.random.random(2)], dtype=np.float32)
        nodes = np.asarray([
            np.random.random(2),
            np.random.random(2)
        ], dtype=np.float32)
        edges = np.asarray(
            [np.random.random(2)],
            dtype=np.float32
        )
        receivers = [0]
        senders = [1]
        n_node = [2]
        n_edge = [1]
        return {
            "graph": gn.graphs.GraphsTuple(
                nodes=nodes,
                edges=edges,
                globals=globals,
                receivers=receivers,
                senders=senders,
                n_node=n_node,
                n_edge=n_edge),
            "mask": np.ones(num_actions)
        }

    @staticmethod
    def make_dummy_graph_tuple():
        """
        Creates fixed test graph input.
        """
        globals = np.asarray([[9.0, 9.0]], dtype=np.float32)
        nodes = np.asarray([
            [1, 2],
            [3, 4]
        ], dtype=np.float32)
        edges = np.asarray(
            [[5.0, 6.0]],
            dtype=np.float32
        )
        receivers = [0]
        senders = [1]
        n_node = [2]
        n_edge = [1]
        return gn.graphs.GraphsTuple(nodes=nodes,
                                     edges=edges,
                                     globals=globals,
                                     receivers=receivers,
                                     senders=senders,
                                     n_node=n_node,
                                     n_edge=n_edge)

    def test_graph_act(self):
        """
        Tests action fetching with individual graphs and batches, non-deterministic and deterministic
        evaluation.
        """
        agent = Agent(
            num_actions=self.NUM_ACTIONS,
        )

        states = {
            "graph": self.make_dummy_graph_tuple(),
            "mask": np.ones(self.NUM_ACTIONS)
        }
        # Test single act.
        actions, _, _ = agent.act(states=states, explore=True)
        print("Action = ", actions)

        # Test batch act.
        batch_states = [states, states]
        actions, _, _ = agent.act(states=batch_states, explore=True)
        print("Batched action = ", actions)

        # Test masked act.
        mask = np.ones(self.NUM_ACTIONS)
        # Zero out action one
        mask[1] = 0

        # Act a few times.
        masked_states = dict(
            graph=self.make_dummy_graph_tuple(),
            mask=mask
        )
        batch_states = [masked_states] * 50
        actions, _, _ = agent.act(states=batch_states, explore=True)

        # Assert there is no 1 in the actions as masked out.
        self.assertNotIn(1, actions)

        # Test deterministic action using explore=False.
        batch_states = [states] * 50
        actions, _, _ = agent.act(states=batch_states, explore=False)

        # All actions should be the same.
        self.assertEqual(len(set(actions)), 1)

    def test_update_functionality(self):
        """
        Tests updates executes, does not test learning itself.
        """
        agent = Agent(
            num_actions=self.NUM_ACTIONS,
        )

        states = {
            "graph": self.make_dummy_graph_tuple(),
            "mask": np.ones(self.NUM_ACTIONS)
        }
        actions, log_probs, value = agent.act(states=states, explore=True)

        # Batch of size 2
        batch_states = [states, states]
        batch_actions = [actions, actions]
        log_probs = [log_probs, log_probs]
        values = [value, value]
        rewards = [1.0, 1.0]

        loss, vf_loss = agent.update(
            states=batch_states,
            actions=batch_actions,
            log_probs=log_probs,
            vf_values=values,
            rewards=rewards,
            terminals=[0.0, 0.0]
        )

        print("Policy Loss = ", loss)
        print("VF loss = ", vf_loss)
        self.assertIsNotNone(loss)
        self.assertIsNotNone(vf_loss)

    def test_learning(self):
        """
        Tests if the update can fit a minimal binary decision from a state - sanity check.
        """
        num_actions = 2
        agent = Agent(
            num_actions=num_actions,
            learning_rate=0.1,
            discount=1.0
        )

        state = self.random_state(num_actions)
        states = []
        actions = []
        log_probs = []
        vf_values = []
        rewards = []
        terminals = []

        # Create a mini batch of data.
        for i in range(32):
            action, log_prob, value = agent.act(states=state, explore=True)

            # Reward 1.0 for action 1, 0.0 for action 0.
            reward = action

            # Append to buffer.
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            vf_values.append(value)
            rewards.append(float(reward))
            terminals.append(True)

        # Run a few updates.
        for _ in range(100):
            loss, vf_loss = agent.update(
                states=states,
                actions=actions,
                log_probs=log_probs,
                vf_values=vf_values,
                rewards=rewards,
                terminals=terminals
            )
            print("Loss = ", loss)

        # Check that the action was learned.
        batch_states = [state] * 50
        actions, _, _ = agent.act(states=batch_states, explore=False)

        # Every action should be 1 -> sums up to len.
        self.assertEqual(sum(actions), len(actions))
