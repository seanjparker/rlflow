import tensorflow as tf
import numpy as np
from graph_nets.graphs import GraphsTuple
from graph_nets import utils_tf


class _BaseAgent(object):
    def __init__(self):
        self.ckpt_manager = None
        self.ckpt = None

    def act(self, *args):
        pass

    def update(self, *args):
        pass

    def save(self):
        """Saves checkpoint to path."""
        path = self.ckpt_manager.save()
        print("Saving model to path = ", path)

    @staticmethod
    def state_action_masked(states, action_xfer):
        if isinstance(states, list):
            tuples = []
            masks = []
            for i, state in enumerate(states):
                xfer_id = int(action_xfer[i])

                xfer_graph_tuple = state["xfers"][xfer_id]
                xfer_graph_tuple = make_eager_graph_tuple(xfer_graph_tuple)
                location_mask = state["location_mask"][xfer_id]

                tuples.append(xfer_graph_tuple)
                masks.append(location_mask)
        else:
            xfer_id = int(action_xfer)

            xfer_graph_tuple = states["xfers"][xfer_id]
            xfer_graph_tuple = make_eager_graph_tuple(xfer_graph_tuple)
            location_mask = states["location_mask"][xfer_id]

            tuples = xfer_graph_tuple
            masks = location_mask

        return tuples, masks

    def load(self):
        """
        Loads model from latest checkpoint. Note: due to eager execution, this can only be called once
        all sonnet modules have been called once, e.g. by executing an act. See example.
        """
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restoring model from path = {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")


def make_eager_graph_tuple(graph_tuple: GraphsTuple):
    """
    Ensures the contents of one or more graph tuples are tf tensors.

    Args:
        graph_tuple (gn.graph.GraphsTuple): GraphsTuple instances.

    Returns:
        Tuple with each value converted to an eager tensor.
    """
    return graph_tuple.replace(
        edges=tf.convert_to_tensor(value=graph_tuple.edges, dtype=tf.float32),
        nodes=tf.convert_to_tensor(value=graph_tuple.nodes, dtype=tf.float32),
        senders=tf.convert_to_tensor(value=graph_tuple.senders, dtype=tf.int32),
        receivers=tf.convert_to_tensor(value=graph_tuple.receivers, dtype=tf.int32),
        n_node=tf.convert_to_tensor(value=graph_tuple.n_node, dtype=tf.int32),
        n_edge=tf.convert_to_tensor(value=graph_tuple.n_edge, dtype=tf.int32),
        globals=tf.convert_to_tensor(value=graph_tuple.globals, dtype=tf.float32)
    )


def zeros_graph(sample_graph, edge_size, node_size, global_size):
    """
    Utility to produce an empty graph.

    Returns:
        A graph with zero-dim features.
    """
    zeros_graphs = sample_graph.replace(nodes=None, edges=None, globals=None)
    zeros_graphs = utils_tf.set_zero_edge_features(zeros_graphs, edge_size)
    zeros_graphs = utils_tf.set_zero_node_features(zeros_graphs, node_size)
    zeros_graphs = utils_tf.set_zero_global_features(zeros_graphs, global_size)
    return zeros_graphs


def discount_all(values, decay, terminal):
    """
    Discounts sequences based on a decay value.

    Args:
        values: Sequence of values to discount.
        decay: Decay value to apply along the sequence.
        terminal: Indicating end of a subsequence.

    Returns:
        An array of shape values with discounted values.
    """
    discounted = []
    i = len(values) - 1
    prev_v = 0.0
    for v in reversed(values):
        # Arrived at new sequence, start over.
        if np.all(terminal[i]):
            prev_v = 0.0

        # Accumulate prior value.
        accum_v = v + decay * prev_v
        discounted.append(accum_v)
        prev_v = accum_v

        i -= 1
    return list(reversed(discounted))


def gae_helper(vf, reward, gamma, gae_lambda, terminals, sequence_indices):
    """
    Computes generalized advantage estimation.
    """
    deltas = []
    # Convert to array for simplicity.
    vf = vf.numpy()
    start_index = 0
    i = 0
    sequence_indices[-1] = True
    for _ in range(len(vf)):
        if np.all(sequence_indices[i]):
            # Compute deltas for this subsequence.
            # Cannot do this all at once because we would need the correct offsets for each sub-sequence.
            vf_slice = list(vf[start_index:i + 1])

            if np.all(terminals[i]):
                vf_slice.append(0)
            else:
                vf_slice.append(vf[i])
            adjusted_v = np.asarray(vf_slice)

            # +1 because we want to include i-th value.
            delta = reward[start_index:i + 1] + gamma * adjusted_v[1:] - adjusted_v[:-1]
            deltas.extend(delta)
            start_index = i + 1
        i += 1

    deltas = np.asarray(deltas)
    return tf.convert_to_tensor(value=discount_all(deltas, gamma * gae_lambda, terminals), dtype=tf.float32)