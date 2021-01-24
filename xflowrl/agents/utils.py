import tensorflow as tf
from graph_nets import utils_tf
import numpy as np


def make_eager_graph_tuple(graph_tuple):
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


def gae_helper(baseline, reward, gamma, gae_lambda, terminals, sequence_indices):
    """
    Computes generalized advantage estimation.
    """
    deltas = []
    # Convert to array for simplicity.
    baseline = baseline.numpy()
    start_index = 0
    i = 0
    sequence_indices[-1] = True
    for _ in range(len(baseline)):
        if np.all(sequence_indices[i]):
            # Compute deltas for this subsequence.
            # Cannot do this all at once because we would need the correct offsets for each sub-sequence.
            baseline_slice = list(baseline[start_index:i + 1])

            if np.all(terminals[i]):
                baseline_slice.append(0)
            else:
                baseline_slice.append(baseline[i])
            adjusted_v = np.asarray(baseline_slice)

            # +1 because we want to include i-th value.
            delta = reward[start_index:i + 1] + gamma * adjusted_v[1:] - adjusted_v[:-1]
            deltas.extend(delta)
            start_index = i + 1
        i += 1

    deltas = np.asarray(deltas)
    return tf.convert_to_tensor(value=discount_all(deltas, gamma * gae_lambda, terminals), dtype=tf.float32)