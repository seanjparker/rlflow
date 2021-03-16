import sonnet as snt
from graph_nets.modules import GraphIndependent
from graph_nets.blocks import NodeBlock, EdgeBlock

from xflowrl.agents.network.layers import Dropout, ReLU, EdgeMapping


def _node_model_fn(node_output_size):
    if node_output_size is None:
        return None
    return lambda: snt.Sequential([snt.Linear(node_output_size), ReLU()])


def _edge_model_fn():
    return snt.Sequential([EdgeMapping()])


def _dropout_fn():
    return snt.Sequential([Dropout()])


class GCN(snt.Module):
    """
    Creates a Graph Convolutional Network that encodes a graph into a latent variable
    """
    def __init__(self, node_output_size, name="gcn"):
        super(GCN, self).__init__(name=name)
        self._network = [
            GraphIndependent(
                node_model_fn=_dropout_fn),
            EdgeBlock(
                edge_model_fn=_edge_model_fn,
                use_edges=False,
                use_receiver_nodes=True,
                use_sender_nodes=False,
                use_globals=False),
            NodeBlock(
                node_model_fn=_node_model_fn(node_output_size),
                use_received_edges=True,
                use_sent_edges=False,
                use_nodes=False,
                use_globals=False)
            ]

    def __call__(self, inputs):
        for layer in self._networks:
            inputs = layer(inputs)
        return inputs
