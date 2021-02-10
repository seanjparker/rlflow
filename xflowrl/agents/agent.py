from xflowrl.agents.models import GraphModel, GraphNetwork
import tensorflow as tf

from xflowrl.agents.utils import make_eager_graph_tuple


class Agent(object):
    """Provides a high level agent API on top of the graph model and documents parameters."""

    def __init__(self, num_actions, discount=0.99, gae_lambda=1.0, reducer=tf.math.unsorted_segment_sum,
                 learning_rate=0.01, vf_learning_rate=0.01, clip_ratio=0.2, num_message_passing_steps=5,
                 policy_layer_size=32, num_policy_layers=2,
                 edge_model_layer_size=8, num_edge_layers=2, node_model_layer_size=8, num_node_layers=2,
                 global_layer_size=8, num_global_layers=2):
        """

        Args:
            num_actions (int): Number of discrete actions to choose from.
            discount (float): Reward discount, typically between 0.95 and 1.0.
            gae_lambda (float): Generalized advantage estimation lambda. See GAE paper.
            reducer (Union[tf.unsorted_segment_sum, tf.unsorted_segment_mean, tf.unsorted_segment_max,
                tf.unsorted_segment_min, tf.unsorted_segment_prod, tf.unsorted_segment_sqrt_n]): Aggregation
                for graph neural network.
            learning_rate (float): Policy learning rate.
            vf_learning_rate (float): Value network learning rate.
            clip_ratio (float): Limits the likelihood ratio between prior and new policy during the update. Does
                not typically require tuning.
            num_message_passing_steps (int): Number of neighbourhood aggregation steps, currently unused - see
                model.
            policy_layer_size (int):  Num policy layers. Also used for value network.
            num_policy_layers (int): Num layers in policy network.
            edge_model_layer_size (int): Hidden layer neurons.
            num_edge_layers (int):  Num layers for edge aggregation MLP.
            node_model_layer_size (int): Hidden layer neurons.
            num_node_layers (int): Num layers for node aggregation MLP.
            global_layer_size (int): Hidden layer neurons.
            num_global_layers (int): Num layers for global aggregation MLP.
        """
        self.main_net = GraphNetwork(
            reducer=reducer,
            edge_model_layer_size=edge_model_layer_size,
            num_edge_layers=num_edge_layers,
            node_model_layer_size=node_model_layer_size,
            num_node_layers=num_node_layers,
            global_layer_size=global_layer_size,
            num_global_layers=num_global_layers
        )

        self.model = GraphModel(
            num_actions=num_actions,
            discount=discount,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            num_message_passing_steps=num_message_passing_steps,
            policy_layer_size=policy_layer_size,
            num_policy_layers=num_policy_layers,
            main_net=self.main_net
        )
        self.pi_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_learning_rate)

        checkpoint_root = "./checkpoint/models"
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), module=self.model,
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

        return self.model.act(states, explore=explore)

    def update(self, states, actions, log_probs, vf_values, rewards, terminals):
        """
        Computes proximal policy updates and value function updates using two separate
        gradient tapes and optimizers.

        Returns:
            loss (float): Policy loss
            vf_loss (float): Value function loss.
        """
        for state in states:
            state["graph"] = make_eager_graph_tuple(state["graph"])

        actions = tf.convert_to_tensor(value=actions)
        log_probs = tf.convert_to_tensor(value=log_probs)
        vf_values = tf.convert_to_tensor(value=vf_values)

        # Eager update mechanism via gradient taping.
        # Note two separate tapes for policy and value net.
        with tf.GradientTape() as tape, tf.GradientTape() as vf_tape:
            pi_loss, vf_loss = self.model.update(states, actions, log_probs, vf_values, rewards, terminals)

        # N.b.: It seems like if a grad is 0, this is interpreted as 'grads do not exist' and throws a warning.
        # the code below filters these out. Comment out to check just in case.
        policy_grads = tape.gradient(pi_loss, self.model.policy_net.trainable_variables)
        policy_grads = [grad if grad is not None else tf.zeros_like(var)
                        for var, grad in zip(self.model.policy_net.trainable_variables, policy_grads)]
        self.pi_optimizer.apply_gradients(zip(policy_grads, self.model.policy_net.trainable_variables))

        value_grads = vf_tape.gradient(vf_loss, self.model.value_net.trainable_variables)
        value_grads = [grad if grad is not None else tf.zeros_like(var)
                       for var, grad in zip(self.model.value_net.trainable_variables, value_grads)]
        self.vf_optimizer.apply_gradients(zip(value_grads, self.model.value_net.trainable_variables))

        # Unpack eager tensor.
        return pi_loss.numpy(), vf_loss.numpy()

    def save(self, path):
        """Saves checkpoint to path."""
        path = self.ckpt_manager.save()
        print("Saving model to path = ", path)

    def load(self, checkpoint_file):
        """
        Loads model from checkpoint. Note: due to eager execution, this can only be called once
        all sonnet modules have been called once, e.g. by executing an act. See example.

        Args:
            checkpoint_file(str): Path to checkpoint.
        """
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restoring model from path = {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")