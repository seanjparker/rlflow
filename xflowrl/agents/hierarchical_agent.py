import os

from xflowrl.agents.models import GraphModel, GraphNetwork
from xflowrl.agents.utils import make_eager_graph_tuple
import tensorflow as tf


class HierarchicalAgent(object):
    """Provides a high level agent API on top of the graph model and documents parameters."""

    def __init__(self, num_actions, num_locations=100,
                 discount=0.99,
                 gae_lambda=1.0,
                 reducer=tf.math.unsorted_segment_sum,
                 learning_rate=0.01,
                 vf_learning_rate=0.01,
                 clip_ratio=0.2,
                 policy_layer_size=32,
                 num_policy_layers=2,
                 edge_model_layer_size=8,
                 num_edge_layers=2,
                 node_model_layer_size=8,
                 num_node_layers=2,
                 global_layer_size=8,
                 num_global_layers=2,
                 message_passing_steps=1,
                 network_name=None,
                 checkpoint_timestamp=None):
        """

        Args:
            num_actions (int): Number of discrete actions to choose from.
            num_locations (int): Number of discrete locations to choose from for each xfer
            discount (float): Reward discount, typically between 0.95 and 1.0.
            gae_lambda (float): Generalized advantage estimation lambda. See GAE paper.
            reducer (Union[tf.unsorted_segment_sum, tf.unsorted_segment_mean, tf.unsorted_segment_max,
                tf.unsorted_segment_min, tf.unsorted_segment_prod, tf.unsorted_segment_sqrt_n]): Aggregation
                for graph neural network.
            learning_rate (float): Policy learning rate.
            vf_learning_rate (float): Value network learning rate.
            clip_ratio (float): Limits the likelihood ratio between prior and new policy during the update. Does
                not typically require tuning.
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
        self.num_actions = num_actions
        self.num_locations = num_locations

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

        # Model is used to predict the xfer that will be applied to the graph
        self.model = GraphModel(
            num_actions=num_actions,
            discount=discount,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            policy_layer_size=policy_layer_size,
            num_policy_layers=num_policy_layers,
            main_net=self.main_net,
            add_noop=True
        )

        # Model is used to predict the location at where the xfer will be applied
        self.sub_model = GraphModel(
            num_actions=num_locations,
            discount=discount,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            policy_layer_size=policy_layer_size,
            num_policy_layers=num_policy_layers,
            main_net=self.main_net,  # Use a different network?
            state_name='xfers',
            mask_name='location_mask',
            reduce_embedding=True
        )

        self.pi_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_learning_rate)

        self.network_name = network_name
        self.checkpoint_timestamp = checkpoint_timestamp

        checkpoint_root = "./checkpoint/models"
        if network_name is not None:
            checkpoint_root += f'/{network_name}'
        if checkpoint_timestamp is not None:
            checkpoint_root += f'/{checkpoint_timestamp}'
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), module=self.model,
                                        optim=self.pi_optimizer, vf_optim=self.vf_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_root, max_to_keep=5)

    def act(self, states, explore=True):
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
            action: An array containing one or more integer actions.
            log_prob: The log likelihood of the action under the current policy. Needs to be stored for update,
                see example.
            vf_value: The value estimate of the vf policy. Needs to be stored for update, see
                example.
        """
        # Convert graph tuples to eager tensors.
        if isinstance(states, list):
            for state in states:
                state["graph"] = make_eager_graph_tuple(state["graph"])
        else:
            states["graph"] = make_eager_graph_tuple(states["graph"])

        main_action, main_logprobs, main_vf_values = self.model.act(states, explore=explore)

        if isinstance(states, list):
            tuples = []
            masks = []
            for i, state in enumerate(states):
                xfer_id = int(main_action[i])

                xfer_graph_tuple = state["xfers"][xfer_id]
                xfer_graph_tuple = make_eager_graph_tuple(xfer_graph_tuple)
                location_mask = state["location_mask"][xfer_id]

                tuples.append(xfer_graph_tuple)
                masks.append(location_mask)
        else:
            xfer_id = int(main_action)

            xfer_graph_tuple = states["xfers"][xfer_id]
            xfer_graph_tuple = make_eager_graph_tuple(xfer_graph_tuple)
            location_mask = states["location_mask"][xfer_id]

            tuples = xfer_graph_tuple
            masks = location_mask

        sub_state = dict(xfers=tuples, location_mask=masks)

        sub_action, sub_logprobs, sub_vf_values = self.sub_model.act(sub_state, explore=explore)

        return main_action, main_logprobs, main_vf_values, sub_action, sub_logprobs, sub_vf_values

    def update(self, states, actions, log_probs, vf_values,
               sub_actions, sub_log_probs, sub_vf_values,
               rewards, terminals):
        """
        Computes proximal policy updates and value function updates using two separate
        gradient tapes and optimizers.

        Returns:
            loss (float): Policy loss
            vf_loss (float): Value function loss.
        """
        for i, state in enumerate(states):
            main_action = actions[i][0]

            state["xfers"] = make_eager_graph_tuple(state["xfers"][main_action])
            state["location_mask"] = state["location_mask"][main_action]
            state["graph"] = make_eager_graph_tuple(state["graph"])

        actions = tf.convert_to_tensor(value=actions)
        log_probs = tf.convert_to_tensor(value=log_probs)
        vf_values = tf.convert_to_tensor(value=vf_values)
        sub_actions = tf.convert_to_tensor(value=sub_actions)
        sub_log_probs = tf.convert_to_tensor(value=sub_log_probs)
        sub_vf_values = tf.convert_to_tensor(value=sub_vf_values)

        # Update the policy and value networks of the xfer predication model
        with tf.GradientTape() as tape, tf.GradientTape() as vf_tape:
            pi_loss, vf_loss, info = self.model.update(states, actions, log_probs, vf_values, rewards, terminals)

        policy_grads = tape.gradient(pi_loss, self.model.policy_net.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(policy_grads, self.model.policy_net.trainable_variables))

        value_grads = vf_tape.gradient(vf_loss, self.model.value_net.trainable_variables)
        self.vf_optimizer.apply_gradients(zip(value_grads, self.model.value_net.trainable_variables))

        # Update the policy and value networks of the location prediction model
        with tf.GradientTape() as sub_tape, tf.GradientTape() as sub_vf_tape:
            sub_policy_loss, sub_vf_loss, _ = self.sub_model.update(states, sub_actions, sub_log_probs, sub_vf_values,
                                                                    rewards, terminals)

        policy_grads = sub_tape.gradient(sub_policy_loss, self.sub_model.policy_net.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(policy_grads, self.sub_model.policy_net.trainable_variables))

        vf_grads = sub_vf_tape.gradient(sub_vf_loss, self.sub_model.value_net.trainable_variables)
        self.vf_optimizer.apply_gradients(zip(vf_grads, self.sub_model.value_net.trainable_variables))

        # Unpack eager tensor.
        return pi_loss.numpy(), vf_loss.numpy(), sub_policy_loss.numpy(), sub_vf_loss.numpy(), info

    def export(self, graph):
        import taso
        import onnx

        onnx_model = taso.export_onnx(graph)
        path = f'./models/{self.network_name}/{self.checkpoint_timestamp}/{self.network_name}.onnx'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        onnx.save(onnx_model, path)

    def save(self):
        """Saves checkpoint to path."""
        path = self.ckpt_manager.save()
        print("Saved model to path = ", path)

    def load(self):
        """
        Loads model from checkpoint. Note: due to eager execution, this can only be called once
        all sonnet modules have been called once, e.g. by executing an act. See example.

        Args:
            checkpoint_file(str): Path to checkpoint.
        """
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restoring model from = {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
