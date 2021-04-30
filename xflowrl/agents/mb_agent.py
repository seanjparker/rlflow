from typing import Union

import numpy as np
import sonnet as snt
import tensorflow as tf

from pathlib import Path

from xflowrl.agents.models import GraphNetwork, GraphModelV2, GraphAEModel, GraphAENetwork
from xflowrl.agents.network.mdrnn import MDRNN, gmm_loss
from xflowrl.agents.utils import make_eager_graph_tuple, _BaseAgent


class RandomAgent(_BaseAgent):
    def __init__(self, batch_size, num_actions, num_locations=100, reducer=tf.math.unsorted_segment_sum,
                 gmm_learning_rate=0.01, message_passing_steps=5,
                 edge_model_layer_size=8, num_edge_layers=2, node_model_layer_size=8, num_node_layers=2,
                 global_layer_size=8, num_global_layers=2, network_name=None, checkpoint_timestamp=None):
        """
        Args:
            num_actions (int): Number of discrete actions to choose from.
            num_locations (int): Number of discrete locations to choose from for each xfer
            reducer (Union[tf.unsorted_segment_sum, tf.unsorted_segment_mean, tf.unsorted_segment_max,
                tf.unsorted_segment_min, tf.unsorted_segment_prod, tf.unsorted_segment_sqrt_n]): Aggregation
                for graph neural network.
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
        self.batch_size = batch_size
        self.latent_size = 32
        self.hidden_size = 256
        self.gaussian_size = 8

        # Create the GNN that will take the dataflow graph as an input and produce an embedding of the graph
        # This is the same as the 'Visual Model (V)' in the World Model paper by Ha et al.
        self.main_net = GraphNetwork(
            reducer=reducer,
            edge_model_layer_size=edge_model_layer_size,
            num_edge_layers=num_edge_layers,
            node_model_layer_size=node_model_layer_size,
            num_node_layers=num_node_layers,
            global_layer_size=self.latent_size,
            num_global_layers=num_global_layers,
            message_passing_steps=message_passing_steps
        )

        # Creates the Mixture Density Recurrent Neural Network that serves as the 'Memory RNN (M)'
        self.mdrnn = MDRNN(batch_size,
                           self.latent_size, num_actions, self.hidden_size, self.gaussian_size, training=True)

        self.trunk = snt.Sequential([self.main_net, self.mdrnn])

        self.model = GraphModelV2(self.trunk, None, batch_size, num_actions)

        self.network_name = network_name
        self.checkpoint_timestamp = checkpoint_timestamp

        self.trunk_lr_fn = tf.optimizers.schedules.PolynomialDecay(gmm_learning_rate, 200, gmm_learning_rate / 40, 2)
        self.trunk_optimizer = tf.optimizers.Adam(self.trunk_lr_fn)

        script_dir = Path(__file__).parent
        relative_path = "../../checkpoint"
        relative_path = (script_dir / relative_path).resolve()

        checkpoint_root = f"{relative_path}/mb/models"
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
                np.concatenate(
                    [self.main_net.get_embeddings(gt["graph"], make_tensor=True).numpy() for gt in batch], axis=0)
            )
        next_latent_state = []
        for batch in next_states:
            next_latent_state.append(
                np.concatenate(
                    [self.main_net.get_embeddings(gt["graph"], make_tensor=True).numpy() for gt in batch], axis=0)
            )

        def pad_and_transpose(t, max_in_dims):
            s = tf.shape(t)
            paddings = [[0, m - s[i]] for i, m in enumerate(max_in_dims)]
            return tf.transpose(tf.pad(t, paddings), [1, 0])

        pad_dims = [self.batch_size, self.hidden_size]
        ragged_shape = [None, self.hidden_size]
        terminals = pad_and_transpose(
            tf.ragged.constant(terminals, dtype=tf.float32).to_tensor(shape=ragged_shape), pad_dims
        )
        rewards = pad_and_transpose(
            tf.ragged.constant(rewards, dtype=tf.float32).to_tensor(shape=ragged_shape), pad_dims
        )

        with tf.GradientTape() as tape:
            (mus, sigmas, log_pi, rs, ds), ns = self.mdrnn(xfer_actions, loc_actions, latent_state)

            scale = self.latent_size + 2
            gmm = gmm_loss(next_latent_state, mus, sigmas, log_pi, num_latents=self.mdrnn.num_latents) / scale

            bce_f = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            bce = bce_f(terminals, ds)

            mse_f = tf.keras.losses.MeanSquaredError()
            mse = mse_f(rewards, rs)

            loss = gmm + bce + mse

            grads = tape.gradient(loss, self.mdrnn.trainable_variables)
            self.trunk_optimizer.apply_gradients(zip(grads, self.mdrnn.trainable_variables))
        return dict(loss=loss, gmm=gmm, bce=bce, mse=mse)

    def update(self, states, next_states, actions, rewards, terminals):
        raise NotImplementedError('Use update_mdrnn instead')


class WorldModelAgent(_BaseAgent):
    def __init__(self, network_name, checkpoint_timestamp, wm_timestamp):
        super().__init__()
        self.network_name = network_name
        self.checkpoint_timestamp = checkpoint_timestamp

        script_dir = Path(__file__).parent
        relative_path = "../../checkpoint"
        checkpoint_root = (script_dir / relative_path).resolve()

        self._ctrl_checkpoint = f"{checkpoint_root}/mb_ctrl/models"
        if network_name is not None:
            self._ctrl_checkpoint += f'/{network_name}'
        if checkpoint_timestamp is not None:
            self._ctrl_checkpoint += f'/{checkpoint_timestamp}'

        self._wm_checkpoint = f"{checkpoint_root}/mb/models"
        if network_name is not None:
            self._wm_checkpoint += f'/{network_name}'
        if wm_timestamp is not None:
            self._wm_checkpoint += f'/{wm_timestamp}'

        self.wm_ckpt = None
        self.wm_ckpt_manager = None

    def act(self, states: tf.Tensor, explore=True):
        raise NotImplementedError

    def update(self, states, xfer_actions, log_probs, vf_values, loc_actions, loc_log_probs, loc_vf_values,
               rewards, terminals):
        raise NotImplementedError

    def load_wm(self):
        assert self.wm_ckpt is not None and self.wm_ckpt_manager is not None

        self.load()
        self.wm_ckpt.restore(self.wm_ckpt_manager.latest_checkpoint)
        if self.wm_ckpt_manager.latest_checkpoint:
            print("Restoring world model from path = {}".format(self.wm_ckpt_manager.latest_checkpoint))
        else:
            raise AssertionError('Failed to load world model checkpoint')


class MBAgent(WorldModelAgent):
    def __init__(self, batch_size, num_actions, num_locations=100, reducer=tf.math.unsorted_segment_sum,
                 pi_learning_rate=0.001, vf_learning_rate=0.001, message_passing_steps=5,
                 edge_model_layer_size=8, num_edge_layers=2,
                 node_model_layer_size=8, num_node_layers=2, global_layer_size=8, num_global_layers=2,
                 network_name=None, checkpoint_timestamp=None, wm_timestamp=None):
        """
        Args:
            num_actions (int): Number of discrete actions to choose from.
            num_locations (int): Number of discrete locations to choose from for each xfer
            reducer (Union[tf.unsorted_segment_sum, tf.unsorted_segment_mean, tf.unsorted_segment_max,
                tf.unsorted_segment_min, tf.unsorted_segment_prod, tf.unsorted_segment_sqrt_n]): Aggregation
                for graph neural network.
            pi_learning_rate (float): Learning rate for the policy network
            vf_learning_rate (float): Learning rate for the value network
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
        super().__init__(network_name, checkpoint_timestamp, wm_timestamp)
        self.batch_size = batch_size
        self.latent_size = 32
        self.hidden_size = 256
        self.gaussian_size = 8

        # Create the GNN that will take the dataflow graph as an input and produce an embedding of the graph
        # This is the same as the 'Visual Model (V)' in the World Model paper by Ha et al.
        self.main_net = GraphNetwork(
            reducer=reducer,
            edge_model_layer_size=edge_model_layer_size,
            num_edge_layers=num_edge_layers,
            node_model_layer_size=node_model_layer_size,
            num_node_layers=num_node_layers,
            global_layer_size=self.latent_size,
            num_global_layers=num_global_layers,
            message_passing_steps=message_passing_steps
        )

        # Creates the Mixture Density Recurrent Neural Network that serves as the 'Memory RNN (M)'
        self.mdrnn = MDRNN(1, self.latent_size, num_actions, self.hidden_size, self.gaussian_size)

        self.trunk = snt.Sequential([self.main_net, self.mdrnn])

        self.model = GraphModelV2(self.trunk, batch_size, num_actions)
        self.sub_model = GraphModelV2(self.trunk, batch_size, num_actions)

        self.pi_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_learning_rate)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_learning_rate)

        self.wm_ckpt = tf.train.Checkpoint(trunk=self.trunk)
        self.wm_ckpt_manager = tf.train.CheckpointManager(self.wm_ckpt, self._wm_checkpoint, max_to_keep=5)

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), pi_optimiser=self.pi_optimizer,
                                        vf_optimiser=self.vf_optimizer, xfer_ctrl=self.model, loc_ctrl=self.sub_model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self._ctrl_checkpoint, max_to_keep=5)

    def act(self, states: tf.Tensor, explore=True):
        """
        Act on one or a list of states.

        Args:
            states (tf.Tensor): A single latent state
            explore (bool): If true, samples an action from the policy according the learned probabilities.
                If false, deterministically uses the maximum likelihood estimate. Set to false during final
                evaluation.
        Returns:
            action: a tuple (xfer, location) that describes the action to perform based on the current state
        """
        xfer_action, xfer_logprobs, xfer_vf_values = self.model.act(states, explore=explore)
        loc_action, loc_logprobs, loc_vf_values = self.sub_model.act(states, explore=explore)

        return xfer_action, xfer_logprobs, xfer_vf_values, loc_action, loc_logprobs, loc_vf_values

    def update(self, states, xfer_actions, log_probs, vf_values, loc_actions, loc_log_probs, loc_vf_values,
               rewards, terminals):
        """
        Computes proximal policy updates and value function updates using two separate
        gradient tapes and optimizers.

        Returns:
            loss (float): Policy loss
            vf_loss (float): Value function loss.
        """

        xfer_actions = tf.convert_to_tensor(value=xfer_actions)
        log_probs = tf.convert_to_tensor(value=log_probs)
        vf_values = tf.convert_to_tensor(value=vf_values)
        loc_actions = tf.convert_to_tensor(value=loc_actions)
        loc_log_probs = tf.convert_to_tensor(value=loc_log_probs)
        loc_vf_values = tf.convert_to_tensor(value=loc_vf_values)

        # Update the policy and value networks of the xfer predication model
        with tf.GradientTape() as tape, tf.GradientTape() as vf_tape:
            pi_loss, vf_loss, info = self.model.update(states, xfer_actions, log_probs, vf_values, rewards, terminals)

            policy_grads = tape.gradient(pi_loss, self.model.policy_net.trainable_variables)
            self.pi_optimizer.apply_gradients(zip(policy_grads, self.model.policy_net.trainable_variables))

            value_grads = vf_tape.gradient(vf_loss, self.model.value_net.trainable_variables)
            self.vf_optimizer.apply_gradients(zip(value_grads, self.model.value_net.trainable_variables))

        # Update the policy and value networks of the location prediction model
        with tf.GradientTape() as sub_tape, tf.GradientTape() as sub_vf_tape:
            loc_policy_loss, loc_vf_loss, _ = self.sub_model.update(states, loc_actions, loc_log_probs, loc_vf_values,
                                                                    rewards, terminals)

            policy_grads = sub_tape.gradient(loc_policy_loss, self.sub_model.policy_net.trainable_variables)
            self.pi_optimizer.apply_gradients(zip(policy_grads, self.sub_model.policy_net.trainable_variables))

            vf_grads = sub_vf_tape.gradient(loc_vf_loss, self.sub_model.value_net.trainable_variables)
            self.vf_optimizer.apply_gradients(zip(vf_grads, self.sub_model.value_net.trainable_variables))

        # Unpack eager tensor.
        return pi_loss.numpy(), vf_loss.numpy(), loc_policy_loss.numpy(), loc_vf_loss.numpy(), info


class MBAgentV2(WorldModelAgent):
    def __init__(self, batch_size, num_actions, num_locations=100, reducer=tf.math.unsorted_segment_sum,
                 pi_learning_rate=0.001, vf_learning_rate=0.001, message_passing_steps=5,
                 edge_model_layer_size=8, num_edge_layers=2,
                 node_model_layer_size=8, num_node_layers=2, global_layer_size=8, num_global_layers=2,
                 network_name=None, checkpoint_timestamp=None, wm_timestamp=None):
        """
        Args:
            num_actions (int): Number of discrete actions to choose from.
            num_locations (int): Number of discrete locations to choose from for each xfer
            reducer (Union[tf.unsorted_segment_sum, tf.unsorted_segment_mean, tf.unsorted_segment_max,
                tf.unsorted_segment_min, tf.unsorted_segment_prod, tf.unsorted_segment_sqrt_n]): Aggregation
                for graph neural network.
            pi_learning_rate (float): Learning rate for the policy network
            vf_learning_rate (float): Learning rate for the value network
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
        super().__init__(network_name, checkpoint_timestamp, wm_timestamp)
        self.batch_size = batch_size
        self.latent_size = 32
        self.hidden_size = 256
        self.gaussian_size = 8

        # Create the GNN that will take the dataflow graph as an input and produce an embedding of the graph
        # This is the same as the 'Visual Model (V)' in the World Model paper by Ha et al.
        self.main_net = GraphAENetwork(
            edge_model_layer_size=edge_model_layer_size,
            node_model_layer_size=node_model_layer_size,
            global_layer_size=self.latent_size,
            message_passing_steps=message_passing_steps
        )

        # Creates the Mixture Density Recurrent Neural Network that serves as the 'Memory RNN (M)'
        self.mdrnn = MDRNN(1, self.latent_size, num_actions, self.hidden_size, self.gaussian_size)

        self.trunk = snt.Sequential([self.main_net, self.mdrnn])

        self.model = GraphModelV2(self.trunk, batch_size, num_actions)
        self.sub_model = GraphModelV2(self.trunk, batch_size, num_actions)

        self.pi_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_learning_rate)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_learning_rate)

        self.wm_ckpt = tf.train.Checkpoint(trunk=self.trunk)
        self.wm_ckpt_manager = tf.train.CheckpointManager(self.wm_ckpt, self._wm_checkpoint, max_to_keep=5)

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), pi_optimiser=self.pi_optimizer,
                                        vf_optimiser=self.vf_optimizer, xfer_ctrl=self.model, loc_ctrl=self.sub_model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self._ctrl_checkpoint, max_to_keep=5)

    def act(self, states: tf.Tensor, explore=True):
        """
        Act on one or a list of states.

        Args:
            states (tf.Tensor): A single latent state
            explore (bool): If true, samples an action from the policy according the learned probabilities.
                If false, deterministically uses the maximum likelihood estimate. Set to false during final
                evaluation.
        Returns:
            action: a tuple (xfer, location) that describes the action to perform based on the current state
        """
        xfer_action, xfer_logprobs, xfer_vf_values = self.model.act(states, explore=explore)
        loc_action, loc_logprobs, loc_vf_values = self.sub_model.act(states, explore=explore)

        return xfer_action, xfer_logprobs, xfer_vf_values, loc_action, loc_logprobs, loc_vf_values

    def update(self, states, xfer_actions, log_probs, vf_values, loc_actions, loc_log_probs, loc_vf_values,
               rewards, terminals):
        """
        Computes proximal policy updates and value function updates using two separate
        gradient tapes and optimizers.

        Returns:
            loss (float): Policy loss
            vf_loss (float): Value function loss.
        """

        xfer_actions = tf.convert_to_tensor(value=xfer_actions)
        log_probs = tf.convert_to_tensor(value=log_probs)
        vf_values = tf.convert_to_tensor(value=vf_values)
        loc_actions = tf.convert_to_tensor(value=loc_actions)
        loc_log_probs = tf.convert_to_tensor(value=loc_log_probs)
        loc_vf_values = tf.convert_to_tensor(value=loc_vf_values)

        # Update the policy and value networks of the xfer predication model
        with tf.GradientTape() as tape, tf.GradientTape() as vf_tape:
            pi_loss, vf_loss, info = self.model.update(states, xfer_actions, log_probs, vf_values, rewards, terminals)

            policy_grads = tape.gradient(pi_loss, self.model.policy_net.trainable_variables)
            self.pi_optimizer.apply_gradients(zip(policy_grads, self.model.policy_net.trainable_variables))

            value_grads = vf_tape.gradient(vf_loss, self.model.value_net.trainable_variables)
            self.vf_optimizer.apply_gradients(zip(value_grads, self.model.value_net.trainable_variables))

        # Update the policy and value networks of the location prediction model
        with tf.GradientTape() as sub_tape, tf.GradientTape() as sub_vf_tape:
            loc_policy_loss, loc_vf_loss, _ = self.sub_model.update(states, loc_actions, loc_log_probs, loc_vf_values,
                                                                    rewards, terminals)

            policy_grads = sub_tape.gradient(loc_policy_loss, self.sub_model.policy_net.trainable_variables)
            self.pi_optimizer.apply_gradients(zip(policy_grads, self.sub_model.policy_net.trainable_variables))

            vf_grads = sub_vf_tape.gradient(loc_vf_loss, self.sub_model.value_net.trainable_variables)
            self.vf_optimizer.apply_gradients(zip(vf_grads, self.sub_model.value_net.trainable_variables))

        # Unpack eager tensor.
        return pi_loss.numpy(), vf_loss.numpy(), loc_policy_loss.numpy(), loc_vf_loss.numpy(), info
