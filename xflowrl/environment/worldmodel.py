from graph_nets import utils_tf
import tensorflow as tf
import tensorflow_probability as tfp
from xflowrl.environment.util import _BaseEnvironment


class WorldModelEnvironment(_BaseEnvironment):
    def __init__(self, num_locations):
        super().__init__(num_locations, False, None)  # num_locations, real_measurements, rwd_func
        self.main_net = None
        self.mdrnn = None
        self.state = None
        self._fully_init = False
        self._interaction_limit = 10

    def init_state(self, main_net, mdrnn):
        self.main_net = main_net
        self.mdrnn = mdrnn
        self._fully_init = True

    def reset_wm(self, graph_state):
        assert self._fully_init is True

        if isinstance(graph_state, list):
            input_list = [state["graph"] for state in graph_state]  # These are e.g. graph tuples
            inputs = utils_tf.concat(input_list, axis=0)
        else:
            inputs = graph_state["graph"]

        self.state = self.main_net.get_embeddings(inputs, make_tensor=True)
        return tf.identity(self.state)

    def step(self, actions):
        assert self._fully_init is True

        if self._interaction_limit == 0:
            # If timeout reached, force terminate
            actions = [[151], [0]]
            self._interaction_limit = 10
        else:
            self._interaction_limit -= 1

        xfer_id, location_id = actions
        print(f'{xfer_id[0]} @ {location_id[0]}')

        (mus, _, log_pi, rs, ds), ns = self.mdrnn(xfer_id, location_id, tf.expand_dims(self.state, axis=0))
        # We are only doing a single step, so extract the first element from the sequence
        mus = tf.squeeze(mus[0, :, :, :])
        log_pi = tf.squeeze(log_pi[0, :, :])
        mixt = tfp.distributions.Categorical(tf.math.exp(log_pi)).sample(1)[0]
        new_state = mus[mixt, :]
        reward = tf.squeeze(rs[0, :])
        terminal = tf.squeeze(ds[0, :] > 0)

        new_state = tf.expand_dims(new_state, axis=0)
        self.state = tf.identity(new_state)

        return new_state, reward.numpy(), terminal.numpy(), None
