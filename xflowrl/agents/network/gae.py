import sonnet as snt
from graph_nets import utils_tf
from graph_nets.modules import GraphIndependent

from xflowrl.agents.network.layers import GraphMLP, GraphMLPNetwork


def _make_edge_fn(edge_output_size):
    if edge_output_size is None:
        return None
    return snt.Linear(edge_output_size, name="edge_output")


def _make_node_fn(node_output_size):
    if node_output_size is None:
        return None
    return snt.Linear(node_output_size, name="node_output")


def _make_global_fn(global_output_size):
    if global_output_size is None:
        return None
    return snt.Linear(global_output_size, name="global_output")


class EncoderDecoder(snt.Module):
    """Encoder-decoder model.
    The model has three components:
    - Encoder graph net that encodes the edge, node, global attributes
    - N rounds of message-passing steps
    - Decoder graph net that decodes the edge, node global attributes on each message-passing step
    """

    def __init__(self,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 name="gae"):
        super(EncoderDecoder, self).__init__(name=name)
        self._encoder = GraphMLP()
        self._core = GraphMLPNetwork()
        self._decoder = GraphMLP()

        edge_fn = _make_edge_fn(edge_output_size)
        node_fn = _make_node_fn(node_output_size)
        global_fn = _make_node_fn(global_output_size)

        self._output_transform = GraphIndependent(edge_fn, node_fn, global_fn)

    def __call__(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            decoded_op = self._decoder(latent)
            output_ops.append(self._output_transform(decoded_op))
        return output_ops
