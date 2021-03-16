import sonnet as snt
import tensorflow as tf
from graph_nets.modules import GraphIndependent, GraphNetwork

NUM_LAYERS = 2
LATENT_SIZE = 16


def make_mlp_model():
    return snt.Sequential([
        snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
        snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
    ])


class GraphMLP(snt.Module):
    """ Graph MLP edge, node, and global models. """

    def __init__(self, name="sonnet_GraphMLP"):
        super(GraphMLP, self).__init__(name=name)
        self._network = GraphIndependent(
            edge_model_fn=make_mlp_model,
            node_model_fn=make_mlp_model,
            global_model_fn=make_mlp_model)

    def __call__(self, inputs):
        return self._network(inputs)


class GraphMLPNetwork(snt.Module):
    """ GraphNetwork with MLP edge, node, and global models. """

    def __init__(self, name="sonnet_GraphMLPNetwork"):
        super(GraphMLPNetwork, self).__init__(name=name)
        self._network = GraphNetwork(make_mlp_model, make_mlp_model, make_mlp_model)

    def __call__(self, inputs):
        return self._network(inputs)


class Dropout(snt.Module):
    def __init__(self, rate=0.1, name='sonnet_Dropout'):
        super(Dropout, self).__init__(name=name)
        self.drop_prob = rate

    def __call__(self, inputs):
        return tf.nn.dropout(inputs, 1 - self.drop_prob, self.name)


class ReLU(snt.Module):
    def __init__(self, name='sonnet_ReLU'):
        super(ReLU, self).__init__(name=name)

    def __call__(self, inputs):
        return tf.nn.relu(inputs, self.name)


class EdgeMapping(snt.Module):
    def __init__(self, name="sonnet_EdgeMapping"):
        super(EdgeMapping, self).__init__(name=name)

    def __call__(self, inputs):
        return inputs
