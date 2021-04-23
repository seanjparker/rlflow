import tensorflow as tf
import numpy as np
import graph_nets as gn
import sonnet as snt
from graph_nets import utils_tf

from xflowrl.agents.network.mdrnn import gmm_loss
from xflowrl.agents.utils import gae_helper, make_eager_graph_tuple


def make_mlp_model(layer_size, num_layers, activate_final=True, activation=tf.nn.relu, name="mlp"):
    """
    Creates Sonnet a MLP used for graph modules.

    Args:
        layer_size (int): Hidden size.
        num_layers (int): Number of layers.
        activate_final (bool): If true, final layer has activation, otherwise final layer is linear.
        activation (tf.nn.activation): Activations for mlp.
    """
    return snt.Sequential([
      snt.nets.MLP([layer_size] * num_layers, activate_final=activate_final, activation=activation),
      snt.LayerNorm(axis=1, create_offset=True, create_scale=True)
    ], name=name)


# TODO: IMPORTANT
# These settings configure how each block aggregation includes graph elements.
# This has to be adjusted based on the problem - what data should be included?

# Further: ensure feature shapes are compatible when combining different graph elements.
# Example: If the edge block aggregates edge features, node features and global features,
# the aggregation will fail if their shapes are incomaptible.
_DEFAULT_EDGE_BLOCK_OPT = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": True,
}

_DEFAULT_NODE_BLOCK_OPT = {
    "use_received_edges": True,
    "use_sent_edges": False,
    "use_nodes": True,
    "use_globals": True,
}

_DEFAULT_GLOBAL_BLOCK_OPT = {
    "use_edges": True,
    "use_nodes": True,
    "use_globals": True,
}


class GraphNetwork(snt.Module):
    def __init__(self,
                 reducer=tf.math.unsorted_segment_sum,
                 edge_model_layer_size=8,
                 num_edge_layers=2,
                 node_model_layer_size=8,
                 num_node_layers=2,
                 global_layer_size=8,
                 num_global_layers=2,
                 message_passing_steps=1):
        super(GraphNetwork, self).__init__()

        self.message_passing_steps = message_passing_steps

        self.graph_net = gn.modules.GraphNetwork(
            edge_model_fn=lambda: make_mlp_model(layer_size=edge_model_layer_size,
                                                 num_layers=num_edge_layers, name="edge_model"),
            node_model_fn=lambda: make_mlp_model(layer_size=node_model_layer_size,
                                                 num_layers=num_node_layers, name="node_model"),
            global_model_fn=lambda: make_mlp_model(layer_size=global_layer_size,
                                                   num_layers=num_global_layers, name="global_model"),
            reducer=reducer,
            edge_block_opt=_DEFAULT_EDGE_BLOCK_OPT,
            node_block_opt=_DEFAULT_NODE_BLOCK_OPT,
            global_block_opt=_DEFAULT_GLOBAL_BLOCK_OPT
        )

    def get_embeddings(self, graph_tuple, make_tensor=False):
        """
        Computes policy logits from graph_tuple. This is done by first computing a graph
        embedding on the incoming graph_tuple.

        Args:
            graph_tuple: Instance of gn.graphs.GraphsTuple.
            make_tensor: Boolean flag to convert the graph_tuple to a tensor if true
        Returns:
            Globals: Tensor of shape graph_tuple.globals.shape
        """
        if make_tensor:
            graph_tuple = make_eager_graph_tuple(graph_tuple)

        # This performs one pass through the graph network and its aggregation functions.
        # In particular, this updates globals by first updating edges based on the edge
        # conditioning, then nodes, then globals.
        updated_graph = self.graph_net(graph_tuple)

        # for _ in range(self.message_passing_steps - 1):
        #    updated_graph = self.graph_net(updated_graph)

        # initial_state = zeros_graph(
        #     graph_tuple, graph_tuple.edges.shape[0], graph_tuple.nodes.shape[0], graph_tuple.globals.shape[0])
        # TODO The exact message passing / aggregation mechanism has to be decided based on problem structure.
        # https://colab.research.google.com/github/deepmind/graph_nets/blob/master/graph_nets/demos/graph_nets_basics.ipynb#scrollTo=bJX9iMMIt8T9
        # This is one example of a recurrent aggregation.

        # Use updated globals as output.
        # Pass result to policy network.
        return updated_graph.globals


class _BaseModel(snt.Module):
    def __init__(self):
        super(_BaseModel, self).__init__()

    @staticmethod
    def masked_logits(logits, mask):
        """
        Masks out invalid actions.

        Args:
            logits: Action logits, shape [B, num_actions]
            mask: Valid actions, shape [B, num_actions]

        Returns:
            masked_logits: Eager tensor of shape [B, num_actions] where all values which were 0 in the corresponding
                mask are now -1e10. This means they will not be sampled when sampling actions from the policy.
        """
        mask = tf.convert_to_tensor(value=mask)
        mask = tf.cast(mask, tf.bool)

        mask_value = tf.cast(
            tf.fill(dims=tf.shape(input=logits), value=-1e10), logits.dtype)
        return tf.where(mask, logits, mask_value)

    def act(self, *args):
        pass

    def update(self, *args):
        pass


class GraphModel(_BaseModel):
    """A graph neural network-based reinforcement learning model in TF eager mode.

    The model implements a one-step proximal policy optimization loss:

    https://arxiv.org/abs/1707.06347

    Networks are implemented using Sonnet and GraphNets. See docs on 'trainable_variables'
    property for comment on versions.
    """

    def __init__(self, num_actions, discount=0.99, gae_lambda=1.0,
                 clip_ratio=0.2, policy_layer_size=32, num_policy_layers=2, num_message_passing_steps=5,
                 main_net=None, state_name='graph', mask_name='mask', reduce_embedding=False, add_noop=False):
        super(GraphModel, self).__init__()

        if add_noop:
            num_actions += 1

        self.num_actions = num_actions
        self.clip_ratio = clip_ratio
        self.discount = discount
        self.gae_lambda = gae_lambda

        # 1. Create graph net module.
        self.main_net = main_net
        self.state_name = state_name
        self.mask_name = mask_name

        self.reduce_embedding = reduce_embedding

        # 2. Create policy net.
        # This is a fully connected layer taking as input the graph embedding output, and outputting
        # one of a discrete set of actions. If another set of actions is required, e.g. for hierarchies,
        # simply add another policy net, and compute its loss via the same update rule as this one.
        self.policy_net = snt.Sequential([
            # Stack of hidden layers.
            snt.nets.MLP([policy_layer_size] * num_policy_layers, activate_final=True, activation=tf.nn.relu),
            # A final layer with num_actions neurons.
            snt.nets.MLP([num_actions], activate_final=False)
        ], name="policy_net")

        # Value function, outputs a value estimate for the current state.
        self.value_net = snt.Sequential([
            snt.nets.MLP([policy_layer_size] * num_policy_layers, activate_final=True, activation=tf.nn.relu),
            snt.nets.MLP([1], activate_final=False)
        ], name="value_net")

    def act(self, states, explore=True):
        """
        Compute actions for states.

        Args:
            states: Input states.
            explore: If true, sample action, if false, use argmax.
        Returns:
            Integer action(s) of dim [B]
            Log-likelihoods for actions [B x NUM_ACTIONS]
        """
        if isinstance(states, list):
            # Concat masks.
            mask = tf.concat([state[self.mask_name] for state in states], axis=0)

            # Convert graph tuple(s) to single tensor graph tuple.
            # Assume batched states are list of graph tuples.
            input_list = [state[self.state_name] for state in states]  # These are e.g. graph tuples
            inputs = utils_tf.concat(input_list, axis=0)
        else:
            inputs = states[self.state_name]
            mask = states[self.mask_name]

        # Separately output globals.
        embedding = self.main_net.get_embeddings(inputs)

        if self.reduce_embedding:
            # In the case of the sub action, we have e.g. 80 possible locations (out of 100). For each of these
            # 80 graphs, we get an embedding of size global_dim. Reduce these information over 80 graphs into one
            # information on which we base our decision on.
            # Todo: Instead pad with zeros and use full embedding matrix?
            embedding = tf.reduce_mean(input_tensor=embedding, axis=0, keepdims=True)

        logits = self.policy_net(embedding)
        vf_values = tf.squeeze(self.value_net(embedding))

        # Logits has dim [B, NUM_ACTIONS]
        # Mask has concat shape [B x NUM_ACTIONS]
        # -> Reshape mask.
        mask = tf.reshape(mask, logits.shape)

        # Mask out invalid actions.
        logits = self.masked_logits(logits, mask)

        if explore:
            action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        else:
            action = tf.convert_to_tensor(value=np.argmax(logits, -1))
        log_probs = tf.nn.log_softmax(logits)
        action_log_prob = tf.reduce_sum(input_tensor=tf.one_hot(tf.squeeze(action), depth=self.num_actions) * log_probs, axis=1)

        # Detach from eager tensor to numpy.
        return action.numpy(), action_log_prob.numpy(), vf_values.numpy()

    def _compute_generalized_advantage_estimate(self, values, rewards, terminals):
        return gae_helper(
            vf=values,
            reward=rewards,
            gamma=self.discount,
            gae_lambda=self.gae_lambda,
            terminals=terminals,
            # Sequence indices can be used if there are multiple sub-sequences from
            # _different_ environment copies which are NOT terminal.
            # If only a single environment is used, this is the same as terminals.
            sequence_indices=terminals
        )

    def update(self, states, actions, prev_log_probs, prev_values, rewards, terminals):
        """
        Computes one proximal policy optimization update step (no-subsampling).

        Returns:
            loss: The policy loss.
        """
        # Transform lists of graph-tuples into one graph tuple containing the entire batch.
        # Also concatenate masks.
        mask = tf.concat([state[self.mask_name] for state in states], axis=0)
        prev_log_probs = tf.squeeze(tf.convert_to_tensor(value=prev_log_probs))
        input_list = [state[self.state_name] for state in states]  # E.g. graph tuples
        if not self.reduce_embedding:
            inputs = utils_tf.concat(input_list, axis=0)
            embedding = self.main_net.get_embeddings(inputs)
        else:
            embedding = []
            # Every graph tuple returns a tuple of shape (NUM_AVAILABLE_LOCATIONS x EMBEDDING_SIZE)
            # We currently cannot concat these in a meaningful way, as each step has a different number
            # of available locations. Thus, we have to run each graph through the network separately.
            # Todo: We might be able to concat these if we use a fixed size (e.g. empty graphs)
            for graph_tuple in input_list:
                graph_embeddings = self.main_net.get_embeddings(graph_tuple)
                graph_embeddings = tf.reduce_mean(input_tensor=graph_embeddings, axis=0, keepdims=False)
                embedding.append(graph_embeddings)

            embedding = tf.convert_to_tensor(value=np.asarray(embedding), dtype=tf.float32)

        logits = self.policy_net(embedding)

        # VF output.
        vf_values = tf.squeeze(self.value_net(embedding))

        # Compute advantage estimate.
        advantages = self._compute_generalized_advantage_estimate(
            values=vf_values,
            rewards=rewards,
            terminals=terminals
        )

        # Compute vf loss with simple mean squared error against prior values.
        v_targets = advantages + prev_values
        v_targets = tf.stop_gradient(input=v_targets)
        vf_loss = (vf_values - v_targets) ** 2

        mask = tf.reshape(mask, logits.shape)
        logits = self.masked_logits(logits, mask)

        # Log likelihood of actions being taken.
        log_probs = tf.nn.log_softmax(logits)
        log_probs = tf.reduce_sum(input_tensor=tf.one_hot(tf.squeeze(actions), depth=self.num_actions) * log_probs, axis=-1)

        # Ratio against log likelihood of actions _before_ update.
        likelihood_ratio = tf.exp(x=log_probs - prev_log_probs)

        # Update is bounded by clip ratio.
        clipped_advantages = tf.where(
            condition=advantages > 0,
            x=((1 + self.clip_ratio) * advantages),
            y=((1 - self.clip_ratio) * advantages)
        )

        # Shape [B], loss per item
        loss = -tf.minimum(x=likelihood_ratio * advantages, y=clipped_advantages)

        # Sample estimates of entropy and KL divergence.
        loss_entropy = tf.reduce_mean(input_tensor=-log_probs)
        kl_divergence = tf.reduce_mean(input_tensor=prev_log_probs - log_probs)

        # Monitor during training. Entropy should decrease over time.
        # KL divergence indicates how much the policy changes each update.
        # Large variations in KL-divergence between updates may indicate too high learning rates.
        # For the toy problems given, these will not be meaningful. For typical benchmarks
        # look for a KL-divergence ~ between 0.01 - 0.05.
        # If things are going well despite larger values, this can be ignored. It can merely help
        # pointing towards a problem if nothing is being learned.
        # print("Loss entropy = {}, KL-divergence between old and new policy = {}".format(loss_entropy, kl_divergence))
        info = dict(loss_entropy=loss_entropy.numpy(), kl_divergence=kl_divergence.numpy())

        # Mean loss across batch, shape [B] -> ()
        return tf.reduce_mean(input_tensor=loss, axis=0), tf.reduce_mean(input_tensor=vf_loss, axis=0), info


class GraphModelV2(_BaseModel):
    """A graph neural network, model-based reinforcement learning model in tensorflow.

        The model implements a World-Model:

        https://arxiv.org/abs/1809.01999

        Networks are implemented using Sonnet and GraphNets. See docs on 'trainable_variables'
        property for comment on versions.
        """

    def __init__(self, trunk, batch_size, num_actions, discount=0.99, gae_lambda=1.0,
                 clip_ratio=0.2, policy_layer_size=32, num_policy_layers=2,
                 state_name='graph', mask_name='mask', reduce_embedding=False):
        super(GraphModelV2, self).__init__()
        self.main_net = trunk._layers[0]
        self.mdrnn = trunk._layers[1]

        self.state_name = state_name
        self.mask_name = mask_name
        self.reduce_embedding = reduce_embedding
        self.clip_ratio = clip_ratio
        self.discount = discount
        self.gae_lambda = gae_lambda

        self.policy_net = snt.Sequential([
            # Stack of hidden layers.
            snt.nets.MLP([policy_layer_size] * num_policy_layers, activate_final=True, activation=tf.nn.relu),
            # A final layer with num_actions neurons.
            snt.nets.MLP([num_actions], activate_final=False)
        ], name="policy_net")

        # Value function, outputs a value estimate for the current state.
        self.value_net = snt.Sequential([
            snt.nets.MLP([policy_layer_size] * num_policy_layers, activate_final=True, activation=tf.nn.relu),
            snt.nets.MLP([1], activate_final=False)
        ], name="value_net")

        self.num_actions = num_actions
        self.batch_size = batch_size

    def _compute_generalized_advantage_estimate(self, values, rewards, terminals):
        return gae_helper(
            vf=values,
            reward=rewards,
            gamma=self.discount,
            gae_lambda=self.gae_lambda,
            terminals=terminals,
            sequence_indices=terminals
        )

    def act(self, states, explore=True):
        """
        Compute actions for states.

        Args:
            states: Input states.
            explore: If true, sample action, if false, use argmax.
        Returns:
            Integer action(s) of dim [B]
        """

        logits = self.policy_net(states)
        vf_values = tf.squeeze(self.value_net(states))

        if explore:
            action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        else:
            action = tf.convert_to_tensor(value=np.argmax(logits, -1))
        log_probs = tf.nn.log_softmax(logits)
        action_log_prob = tf.reduce_sum(
            input_tensor=tf.one_hot(tf.squeeze(action), depth=self.num_actions) * log_probs, axis=1
        )

        return action.numpy(), action_log_prob.numpy(), vf_values.numpy()

    def update(self, states, actions, prev_log_probs, prev_values, rewards, terminals):
        prev_log_probs = tf.squeeze(tf.convert_to_tensor(value=prev_log_probs))

        logits = self.policy_net(states)

        # VF output.
        vf_values = tf.squeeze(self.value_net(states))

        # Compute advantage estimate.
        advantages = self._compute_generalized_advantage_estimate(
            values=vf_values,
            rewards=rewards,
            terminals=terminals
        )

        # Compute vf loss with simple mean squared error against prior values.
        v_targets = advantages + prev_values
        v_targets = tf.stop_gradient(input=v_targets)
        vf_loss = (vf_values - v_targets) ** 2

        # Log likelihood of actions being taken.
        log_probs = tf.nn.log_softmax(logits)
        log_probs = tf.reduce_sum(
            input_tensor=tf.one_hot(tf.squeeze(actions), depth=self.num_actions) * log_probs, axis=-1
        )

        # Ratio against log likelihood of actions _before_ update.
        likelihood_ratio = tf.exp(x=log_probs - prev_log_probs)

        # Update is bounded by clip ratio.
        clipped_advantages = tf.where(
            condition=advantages > 0,
            x=((1 + self.clip_ratio) * advantages),
            y=((1 - self.clip_ratio) * advantages)
        )

        # Shape [B], loss per item
        loss = -tf.minimum(x=likelihood_ratio * advantages, y=clipped_advantages)

        # Sample estimates of entropy and KL divergence.
        loss_entropy = tf.reduce_mean(input_tensor=-log_probs)
        kl_divergence = tf.reduce_mean(input_tensor=prev_log_probs - log_probs)

        info = dict(loss_entropy=loss_entropy.numpy(), kl_divergence=kl_divergence.numpy())

        # Mean loss across batch, shape [B] -> ()
        return tf.reduce_mean(input_tensor=loss, axis=0), tf.reduce_mean(input_tensor=vf_loss, axis=0), info


class GraphAEModel(_BaseModel):
    def __init__(self, *args, **kwargs):
        super(GraphAEModel, self).__init__()

    def act(self, states, explore=True):
        pass

    def update(self, states, actions, rewards, terminals):
        pass


class RandomGraphModel(_BaseModel):
    """A graph neural network-based reinforcement learning model in TF eager mode.

    The model implements a one-step proximal policy optimization loss:

    https://arxiv.org/abs/1707.06347

    Networks are implemented using Sonnet and GraphNets. See docs on 'trainable_variables'
    property for comment on versions.
    """

    def __init__(self, num_actions, discount=0.99, gae_lambda=1.0,
                 clip_ratio=0.2, policy_layer_size=32, num_policy_layers=2, num_message_passing_steps=5,
                 main_net=None, state_name='graph', mask_name='mask', reduce_embedding=False, add_noop=False):
        super(RandomGraphModel, self).__init__()

        if add_noop:
            num_actions += 1

        self.num_actions = num_actions
        self.clip_ratio = clip_ratio
        self.discount = discount
        self.gae_lambda = gae_lambda

        # 1. Create graph net module.
        self.main_net = main_net
        self.state_name = state_name
        self.mask_name = mask_name

        self.reduce_embedding = reduce_embedding

    def act(self, states, explore=True):
        """
        Compute actions for states.

        Args:
            states: Input states.
            explore: If true, sample action, if false, use argmax.
        Returns:
            Integer action(s) of dim [B]
        """
        if isinstance(states, list):
            # Concat masks.
            mask = tf.concat([state[self.mask_name] for state in states], axis=0)

            # Convert graph tuple(s) to single tensor graph tuple.
            # Assume batched states are list of graph tuples.
            input_list = [state[self.state_name] for state in states]  # These are e.g. graph tuples
            inputs = utils_tf.concat(input_list, axis=0)
        else:
            inputs = states[self.state_name]
            mask = states[self.mask_name]

        # Separately output globals.
        embedding = self.main_net.get_embeddings(inputs)

        if self.reduce_embedding:
            # In the case of the sub action, we have e.g. 80 possible locations (out of 100). For each of these
            # 80 graphs, we get an embedding of size global_dim. Reduce these information over 80 graphs into one
            # information on which we base our decision on.
            # Todo: Instead pad with zeros and use full embedding matrix?
            embedding = tf.reduce_mean(input_tensor=embedding, axis=0, keepdims=True)

        logits = self.policy_net(embedding)
        mask = tf.reshape(mask, logits.shape)
        logits = self.masked_logits(logits, mask)

        action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

        # Detach from eager tensor to numpy.
        return action.numpy()

    def update(self, states, actions, prev_log_probs, prev_values, rewards, terminals):
        raise NotImplementedError
