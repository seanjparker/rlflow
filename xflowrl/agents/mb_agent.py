from typing import Union

import sonnet as snt
import tensorflow as tf

from xflowrl.agents.models import GraphNetwork, GraphModelV2
from xflowrl.agents.network.controller import Controller
from xflowrl.agents.network.mdrnn import MDRNN, gmm_loss
from xflowrl.agents.utils import make_eager_graph_tuple, _BaseAgent


class RandomAgent(_BaseAgent):
    def __init__(self, batch_size, num_actions, num_locations=100, reducer=tf.math.unsorted_segment_sum,
                 controller_learning_rate=0.001, gmm_learning_rate=0.01, message_passing_steps=5,
                 edge_model_layer_size=8, num_edge_layers=2, node_model_layer_size=8, num_node_layers=2,
                 global_layer_size=8, num_global_layers=2, network_name=None, checkpoint_timestamp=None):
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
        self.mdrnn = MDRNN(batch_size, self.latent_size, num_actions, 256, 8)

        self.trunk = snt.Sequential([self.main_net, self.mdrnn])

        self.model = GraphModelV2(self.trunk, None, batch_size, num_actions)

        self.network_name = network_name
        self.checkpoint_timestamp = checkpoint_timestamp

        self.trunk_optimizer = tf.keras.optimizers.RMSprop(learning_rate=gmm_learning_rate)

        checkpoint_root = "./checkpoint/mb/models"
        if network_name is not None:
            checkpoint_root += f'/{network_name}'
        if checkpoint_timestamp is not None:
            checkpoint_root += f'/{checkpoint_timestamp}'
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), trunk=self.trunk, trunk_optimizer=self.trunk_optimizer)
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

        def logical_mask(mask_name="mask", mask=None):
            mask = states[mask_name] if mask is None else mask
            values = tf.cast(tf.convert_to_tensor(value=mask), tf.bool)
            return tf.where(values)

        def random_choice(x, size=1):
            if x.shape[0] > 1:
                # 20% chance of picking the terminating action under normal conditions
                a_prob = 0.8 / (x.shape[0] - 1)
                probabilities = tf.constant(([a_prob] * (x.shape[0] - 1)) + [0.2])
            else:
                probabilities = tf.constant([1.0])
            rescaled_probs = tf.expand_dims(tf.math.log(probabilities), 0)
            idx = tf.squeeze(tf.random.categorical(rescaled_probs, num_samples=size), axis=[0])
            return tf.gather(x, idx)

        xfer_action = random_choice(tf.squeeze(logical_mask(), axis=-1))

        _, location_mask = self.state_xfer_masked(states, xfer_action)
        location_action = random_choice(tf.squeeze(logical_mask(mask=location_mask), axis=1))

        return xfer_action.numpy(), location_action.numpy()

    def update_mdrnn(self, states, next_states, xfer_actions, loc_actions, terminals, rewards):
        latent_state = []
        for batch in states:
            latent_state.append(
                tf.concat([self.main_net.get_embeddings(gt["graph"], make_tensor=True) for gt in batch], axis=0)
            )

        next_latent_state = []
        for batch in next_states:
            next_latent_state.append(
                tf.concat([self.main_net.get_embeddings(gt["graph"], make_tensor=True) for gt in batch], axis=0)
            )

        with tf.GradientTape() as tape:
            (mus, sigmas, log_pi, rs, ds), ns = self.mdrnn(xfer_actions, loc_actions,
                                                           latent_state, self.model.mdrnn_state)
            gmm = gmm_loss(next_latent_state, mus, sigmas, log_pi, num_latents=self.mdrnn.num_latents)
            # bce_f = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            # bce = bce_f(terminals, ds)

            # mse = tf.keras.losses.mse(rewards, rs)
            # scale = self.latent_size + 2
            scale = self.latent_size
            # loss = (gmm + bce + mse) / scale
            loss = gmm / scale

            grads = tape.gradient(loss, self.mdrnn.trainable_variables)
            self.trunk_optimizer.apply_gradients(zip(grads, self.mdrnn.trainable_variables))
        # Store state for next forward pass
        self.model.mdrnn_state = ns
        return loss

    def update(self, states, next_states, actions, rewards, terminals):
        raise NotImplementedError('Use update_mdrnn instead')


class MBAgent(_BaseAgent):
    def __init__(self, batch_size, num_actions, num_locations=100, reducer=tf.math.unsorted_segment_sum,
                 controller_learning_rate=0.001, message_passing_steps=5,
                 edge_model_layer_size=8, num_edge_layers=2,
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
        self.mdrnn = MDRNN(batch_size, self.latent_size, num_actions, 256, 8)

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

        self.ctrl_optimiser = tf.keras.optimizers.Adam(learning_rate=controller_learning_rate)

        checkpoint_root = "./checkpoint/mb_ctrl/models"
        if network_name is not None:
            checkpoint_root += f'/{network_name}'
        if checkpoint_timestamp is not None:
            checkpoint_root += f'/{checkpoint_timestamp}'
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), trunk=self.trunk, ctrl_optimiser=self.ctrl_optimiser,
                                        xfer_ctrl=self.xfer_controller, loc_ctrl=self.loc_controller)
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

        xfer_action = tf.Tensor()

        _, location_mask = self.state_xfer_masked(states, xfer_action)
        location_action = tf.Tensor()

        return xfer_action.numpy(), location_action.numpy()

    def update(self, states, next_states, actions, rewards, terminals):
        for state in states:
            state["graph"] = make_eager_graph_tuple(state["graph"])
        for state in next_states:
            state["graph"] = make_eager_graph_tuple(state["graph"])
        actions = tf.convert_to_tensor(value=actions)

        # # Train head controller
        # with tf.GradientTape() as controller_tape:
        #     xfer_loss = controller_loss()
        # grads = controller_tape.gradient(xfer_loss, self.xfer_controller.trainable_variables)
        # self.con_optimizer.apply_gradients(zip(grads, self.xfer_controller.trainable_variables))
        xfer_loss = tf.Tensor()  # Temp
        return xfer_loss.numpy()
