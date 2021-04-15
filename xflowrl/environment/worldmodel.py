from xflowrl.environment.util import _BaseEnvironment


class WorldModelEnvironment(_BaseEnvironment):
    def __init__(self, main_net, mdrnn, num_locations, reward_function):
        super().__init__(num_locations, False, reward_function)  # num_locations, real_measurements, rwd_func
        self.main_net = main_net
        self.mdrnn = mdrnn
        self.mdrnn_state = self.mdrnn.initial_state()
        self.latent_state = None

    def step(self, actions):
        xfer_id, location_id = actions
        print(f'{xfer_id[0]} @ {location_id[0]}')

        (mus, sigmas, log_pi, rs, ds), ns = self.mdrnn(xfer_id, location_id,
                                                       self.latent_state, self.mdrnn_state)

        # return new_state, reward, terminal, None
        return None
