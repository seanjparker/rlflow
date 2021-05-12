import time

from graph_nets import utils_tf
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
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

        # self._interaction_limit = 10

        if isinstance(graph_state, list):
            input_list = [state["graph"] for state in graph_state]  # These are e.g. graph tuples
            inputs = utils_tf.concat(input_list, axis=0)
        else:
            inputs = graph_state["graph"]

        self.state = self.main_net.get_embeddings(inputs, make_tensor=True)
        return tf.identity(self.state)

    def step(self, actions, step_real=True):
        assert self._fully_init is True

        # if self._interaction_limit == 0:
        #     # If timeout reached, force terminate
        #     actions = np.array([[151], [0]])
        #     self._interaction_limit = 10
        # else:
        #     self._interaction_limit -= 1

        xfer_id, location_id = actions
        # print(f'{xfer_id[0]} @ {location_id[0]}')

        xfer_mask, loc_mask, real_reward, real_terminate = self._step_real(xfer_id, location_id)

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

        return new_state, reward.numpy(), terminal.numpy() or real_terminate, \
            dict(xfer_mask=xfer_mask, loc_mask=loc_mask, real_reward=real_reward, real_terminate=real_terminate)

    def _step_real(self, xfer_id, location_id) -> (tf.Tensor, tf.Tensor, float, bool):
        terminate = False
        if xfer_id >= self.rl_opt.get_num_xfers():
            new_graph = self.graph
            terminate = True
        else:
            new_graph = self.rl_opt.apply_xfer(xfer_id, location_id)

        if new_graph:
            self.graph = new_graph
            costs_dict = self.get_detailed_costs()
            reward = costs_dict['runtime']
        else:
            reward = -1000.0
        new_real_env_state = self.build_state()

        terminal = False
        if np.sum(new_real_env_state['mask']) == 0 or terminate:
            terminal = True

        return new_real_env_state['mask'], new_real_env_state['location_mask'], reward, terminal
