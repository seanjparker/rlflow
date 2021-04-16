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
        return self.state

    def step(self, actions):
        assert self._fully_init is True

        xfer_id, location_id = actions
        print(f'{xfer_id[0]} @ {location_id[0]}')

        (mus, _, log_pi, rs, ds), ns = self.mdrnn(xfer_id, location_id, tf.expand_dims(self.state, axis=0))
        mixt = tfp.distributions.Categorical(tf.math.exp(log_pi)).sample(1)
        new_state = mus[:, mixt, :]
        reward = rs
        terminal = ds > 0

        self.state = new_state

        return new_state, reward, terminal, None
