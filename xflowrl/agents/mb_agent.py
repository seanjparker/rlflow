from xflowrl.agents.models import GraphModel, GraphNetwork, GraphModelV2
import tensorflow as tf

from xflowrl.agents.network.controller import Controller
from xflowrl.agents.network.mdrnn import MDRNN
from xflowrl.agents.utils import make_eager_graph_tuple, _BaseAgent


class MBAgent(_BaseAgent):
    def __init__(self, num_actions, num_locations=100, reducer=tf.math.unsorted_segment_sum,
                 learning_rate=0.01, num_message_passing_steps=5,
                 policy_layer_size=32, num_policy_layers=2, edge_model_layer_size=8, num_edge_layers=2,
                 node_model_layer_size=8, num_node_layers=2, global_layer_size=8, num_global_layers=2):
        """

        Args:
            num_actions (int): Number of discrete actions to choose from.
            discount (float): Reward discount, typically between 0.95 and 1.0.
            gae_lambda (float): Generalized advantage estimation lambda. See GAE paper.
            reducer (Union[tf.unsorted_segment_sum, tf.unsorted_segment_mean, tf.unsorted_segment_max,
                tf.unsorted_segment_min, tf.unsorted_segment_prod, tf.unsorted_segment_sqrt_n]): Aggregation
                for graph neural network.
            learning_rate (float): Learning rate.
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
            num_global_layers=num_global_layers
        )

        # Creates the Mixture Density Recurrent Neural Network that serves as the 'Memory RNN (M)'
        self.mdrnn = MDRNN(self.main_net.globals.shape, num_actions, 256, 5)

        # The controller is an MLP that uses the latent variables from the GNN and MDRNN as inputs
        # it returns a single tensor of size [B, num_xfers] for the xfers
        self.xfer_controller = Controller(num_actions)

        # The location controller chooses the location at which to apply the chosen xfer of size [B, num_locations]
        self.loc_controller = Controller(num_locations)

        self.model = GraphModelV2(self.main_net, self.mdrnn, self.xfer_controller)
        self.sub_model = GraphModelV2(self.main_net, self.mdrnn, self.loc_controller)

        checkpoint_root = "./checkpoint/models"
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=self.model, sub_model=self.sub_model)
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
            action: a tuple (xfer, location) that describes the action to perform based on the current state
        """
        # Convert graph tuples to eager tensors.
        if isinstance(states, list):
            for state in states:
                state["graph"] = make_eager_graph_tuple(state["graph"])
        else:
            states["graph"] = make_eager_graph_tuple(states["graph"])

        xfer_action = self.model.act(states, explore=explore)

        tuples, masks = self.state_xfer_masked(states, xfer_action)
        masked_state = dict(xfers=tuples, location_mask=masks)

        loc_action = self.sub_model.act(masked_state, explore=explore)

        return xfer_action, loc_action

    def update(self, states, actions, log_probs, vf_values, rewards, terminals):
        return None