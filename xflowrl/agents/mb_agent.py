from typing import Union

import sonnet as snt
import tensorflow as tf
import graph_nets as gn
import numpy as np

from xflowrl.agents.models import GraphNetwork, GraphModelV2
from xflowrl.agents.network.controller import Controller
from xflowrl.agents.network.mdrnn import MDRNN
from xflowrl.agents.utils import make_eager_graph_tuple, _BaseAgent


class MBAgent(_BaseAgent):
    def __init__(self, batch_size, num_actions, num_locations=100, reducer=tf.math.unsorted_segment_sum,
                 controller_learning_rate=0.001, gmm_learning_rate=0.01, message_passing_steps=5,
                 policy_layer_size=32, num_policy_layers=2, edge_model_layer_size=8, num_edge_layers=2,
                 node_model_layer_size=8, num_node_layers=2, global_layer_size=8, num_global_layers=2,
                 network_name=None, checkpoint_timestamp=None):
        """

        Args:
            num_actions (int): Number of discrete actions to choose from.
            num_locations (int): Number of discrete locations to choose from for each xfer
            reducer (Union[tf.unsorted_segment_sum, tf.unsorted_segment_mean, tf.unsorted_segment_max,
                tf.unsorted_segment_min, tf.unsorted_segment_prod, tf.unsorted_segment_sqrt_n]): Aggregation
                for graph neural network.
            controller_learning_rate (float): Learning rate for the controllers
            gmm_learning_rate (float): Learning rate for MDRNN
            message_passing_steps (int): Number of neighbourhood aggregation steps, currently unused - see
                model.
            policy_layer_size (int):  Num policy layers. Also used for value network.
            num_policy_layers (int): Num layers in policy network.
            edge_model_layer_size (int): Hidden layer neurons.
            num_edge_layers (int):  Num layers for edge aggregation MLP.
            node_model_layer_size (int): Hidden layer neurons.
            num_node_layers (int): Num layers for node aggregation MLP.
            global_layer_size (int): Hidden layer neurons.
            num_global_layers (int): Num layers for global aggregation MLP.
            network_name (str): Name of the network that is being optimized.
            checkpoint_timestamp (str): Timestamp for continuing the training of an existing model.
        """
        super().__init__()

        # Create the GNN that will take the dataflow graph as an input and produce an embedding of the graph
        # This is the same as the 'Visual Model (V)' in the World Model paper by Ha et al.
        self.main_net = GraphNetwork(
            reducer=reducer,
            edge_model_layer_size=edge_model_layer_size,
            num_edge_layers=num_edge_layers,
            node_model_layer_size=node_model_layer_size,
            num_node_layers=num_node_layers,
            global_layer_size=global_layer_size,
            num_global_layers=num_global_layers,
            message_passing_steps=message_passing_steps
        )

        # Creates the Mixture Density Recurrent Neural Network that serves as the 'Memory RNN (M)'
        self.latent_size = num_locations * global_layer_size
        self.mdrnn = MDRNN(batch_size, self.latent_size, num_actions, 256, 5)

        # The controller is an MLP that uses the latent variables from the GNN and MDRNN as inputs
        # it returns a single tensor of size [B, num_xfers] for the xfers
        self.xfer_controller = Controller(num_actions)

        # The location controller chooses the location at which to apply the chosen xfer of size [B, num_locations]
        self.loc_controller = Controller(num_locations)

        self.trunk = snt.Sequential([self.main_net, self.mdrnn])

        self.model = GraphModelV2(self.trunk, self.xfer_controller, batch_size, num_actions)
        self.sub_model = GraphModelV2(self.trunk, self.loc_controller, batch_size, num_actions)

        self.network_name = network_name
        self.checkpoint_timestamp = checkpoint_timestamp

        self.trunk_optimizer = tf.keras.optimizers.RMSprop(learning_rate=gmm_learning_rate)
        self.con_optimizer = tf.keras.optimizers.Adam(learning_rate=controller_learning_rate)

        checkpoint_root = "./checkpoint/models"
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=self.model, sub_model=self.sub_model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_root, max_to_keep=5)

    def act(self, states: Union[dict, list], explore=True):
        """
        Act on one or a list of states.

        Args:
            states (Union[dict, list]): Either a single state or a list of states.
                Each state is a dict which must contain the keys:
                    'graph': An instance of gn.graphs.GraphTuple
                    'mask': A 1d array of len 'num_actions' indicating valid and invalid actions.
                        Valid actions are 1, invalid actions are 0. At least one action must be valid.
            explore (bool): If true, samples an action from the policy according the learned probabilities.
                If false, deterministically uses the maximum likelihood estimate. Set to false during final
                evaluation.
        Returns:
            action: a tuple (xfer, location) that describes the action to perform based on the current state
        """
        # Convert graph tuples to eager tensors.
        if isinstance(states, list):
            for state in states:
                state["graph"] = make_eager_graph_tuple(state["graph"])
        else:
            states["graph"] = make_eager_graph_tuple(states["graph"])

        def logical_mask():
            mask = states["mask"]
            values = tf.cast(tf.convert_to_tensor(value=mask), tf.bool)
            return tf.where(values)

        def random_choice(x, size, axis=0):
            dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
            indices = tf.range(0, dim_x, dtype=tf.int64)
            sample_index = tf.random.shuffle(indices)[:size]
            sample = tf.gather(x, sample_index, axis=axis)
            return sample

        xfer_action = random_choice(tf.squeeze(logical_mask(), axis=-1), 1)

        tuples, masks = self.state_xfer_masked(states, xfer_action)
        masked_state = dict(xfers=tuples, location_mask=masks)

        loc_action = self.sub_model.act(masked_state, explore=explore)

        return xfer_action, loc_action

    def update(self, states, next_states, actions, rewards, terminals):
        for state in states:
            state["graph"] = make_eager_graph_tuple(state["graph"])
        for state in next_states:
            state["graph"] = make_eager_graph_tuple(state["graph"])
        actions = tf.convert_to_tensor(value=actions)

        # Train network trunk
        with tf.GradientTape() as tape:
            trunk_loss = self.model.update(states, actions, rewards, next_states, terminals)
        grads = tape.gradient(trunk_loss, self.trunk.trainable_variables)
        self.trunk_optimizer.apply_gradients(zip(grads, self.trunk.trainable_variables))

        # Train head controller
        with tf.GradientTape() as controller_tape:
            xfer_loss = controller_loss()
        grads = controller_tape.gradient(xfer_loss, self.xfer_controller.trainable_variables)
        self.con_optimizer.apply_gradients(zip(grads, self.xfer_controller.trainable_variables))

        return trunk_loss.numpy(), xfer_loss.numpy()
