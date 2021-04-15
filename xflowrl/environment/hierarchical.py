import numpy as np

from xflowrl.environment.util import _BaseEnvironment


class HierarchicalEnvironment(_BaseEnvironment):
    def __init__(self, num_locations=100, real_measurements=False, reward_function=None):
        super().__init__(num_locations, real_measurements, reward_function)

    def step(self, actions):
        xfer_id, location_id = actions
        print(f'{xfer_id[0]} @ {location_id[0]}')

        terminate = False
        if xfer_id >= self.rl_opt.get_num_xfers():
            # No-op action terminates the sequence
            new_graph = self.graph
            terminate = True
        else:
            new_graph = self.rl_opt.apply_xfer(xfer_id, location_id)

        if new_graph:
            self.graph = new_graph
            # Make sure to only use the estimated cost before final eval
            # new_run_time = self.get_cost(real_measurement=False)
            costs_dict = self.get_detailed_costs()

            if self.custom_reward is not None:
                for k, v in costs_dict.items():
                    costs_dict[k] = self._normalize_measurements(k, v)
                reward = self.custom_reward(self.last_runtime, costs_dict)
            else:
                reward = costs_dict['runtime']  # Incremental reward

            # reward = 0.  # End-of-episode reward
            self.last_runtime = costs_dict['runtime']
        else:
            print("Invalid action: xfer {} with location {}".format(xfer_id, location_id))
            new_run_time = 0.
            reward = -1000.

        new_state = self.build_state()

        terminal = False
        if np.sum(new_state['mask']) == 0 or terminate:
            # reward = self.initial_runtime - new_run_time  # End-of-episode reward
            terminal = True

        return new_state, reward, terminal, None

    def _normalize_measurements(self, x_key, x):
        if x_key not in self.measurement_info:
            self.measurement_info[x_key] = []
        self.measurement_info[x_key].append(x)

        val = self.measurement_info[x_key]
        x_min, x_max = np.min(val), np.max(val)
        return 2 * ((x - x_min) / (x_max - x_min + 1e-6)) - 1
