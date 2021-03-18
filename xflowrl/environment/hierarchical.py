from functools import partial

import numpy as np
import graph_nets as gn

from xflowrl.core import PyRLOptimizer

from xflowrl.environment.util import graph_to_graphnet_tuple


class _BaseEnvironment(object):
    def __init__(self, num_locations, real_measurements, reward_function):
        """
        Args:
            num_locations (int): number of possible locations to apply Graph Xfers
            real_measurements (bool): If True, the environment will perform real-time measurements of the graph
            reward_function (Callable): A custom reward function can
                be provided with the signature, (last_runtime, new_runtime) -> reward
        """
        self.graph = None
        self.rl_opt = None

        self.locations = None
        self.xfer_graphs = None
        # self.xfer_inputs = None

        self.initial_runtime = 0.0
        self.last_runtime = 0.0
        self.measurement_info = dict()
        self.num_locations = num_locations

        self.real_measurements = real_measurements
        self.custom_reward = reward_function

    def set_graph(self, graph):
        self.graph = graph
        self.rl_opt = PyRLOptimizer(graph)

    def get_cost(self, real_measurement=None):
        if self.real_measurements or real_measurement:
            return self.rl_opt.get_measured_runtime(self.graph)
        else:
            return self.rl_opt.get_cost()

    def get_detailed_costs(self):
        new_run_time, flops, mem_acc, num_kernels = self.graph.get_costs()
        costs_dict = dict(runtime=new_run_time, flops=flops, mem_acc=mem_acc, num_kernels=num_kernels)
        return costs_dict

    def reset(self):
        if not self.rl_opt:
            raise ValueError("Set graph first.")
        self.rl_opt.reset()
        self.last_runtime = self.initial_runtime = self.get_cost()
        return self.build_state()

    def build_state(self):
        self.locations = self.rl_opt.get_available_locations()
        self.xfer_graphs = self.rl_opt.get_xfer_graphs()
        # self.xfer_inputs = self.rl_opt.get_xfer_inputs()

        # Xfer mask
        xfer_mask = np.asarray(self.locations).astype(bool).astype(int)
        xfer_mask = np.append(xfer_mask, 1)

        # Main graphnet tuple
        graph_tuple = graph_to_graphnet_tuple(self.graph, op_runtime_callback=self.rl_opt.get_op_runtime)

        # Sub graphnet tuple
        xfer_tuples = []
        location_masks = []
        for xfer in self.xfer_graphs + [[self.graph]]:  # [[self.graph]] is the dummy no-op embedding
            sum_n_nodes = 0
            sum_n_edges = 0

            init = True
            total_nodes = []
            total_edges = []
            total_globals = []
            total_receivers = []
            total_senders = []
            total_n_nodes = []
            total_n_edges = []

            for xg in xfer:
                nodes, edges, globals, receivers, senders, n_node, n_edge = graph_to_graphnet_tuple(
                    xg,
                    op_runtime_callback=partial(self.rl_opt.get_op_runtime_for_graph, xg),
                    return_graphstuple=False,
                    start_n_node=sum_n_nodes,
                    start_n_edge=sum_n_edges
                )

                if init:
                    total_nodes = nodes
                    total_edges = edges
                    total_globals = globals
                    total_receivers = receivers
                    total_senders = senders
                    total_n_nodes = n_node
                    total_n_edges = n_edge
                    init = False
                else:

                    total_nodes = np.concatenate([total_nodes, nodes], axis=0)
                    total_edges = np.concatenate([total_edges, edges], axis=0)
                    total_globals = np.concatenate([total_globals, globals], axis=0)
                    total_receivers = np.concatenate([total_receivers, receivers], axis=0)
                    total_senders = np.concatenate([total_senders, senders], axis=0)
                    total_n_nodes += n_node
                    total_n_edges += n_edge

                sum_n_nodes += n_node[0]
                sum_n_edges += n_edge[0]

            if (isinstance(total_nodes, list) and total_nodes) or (isinstance(total_nodes, np.ndarray) and total_nodes.any()):
                this_xfer_tuple = gn.graphs.GraphsTuple(
                    nodes=total_nodes,
                    edges=total_edges,
                    globals=total_globals,
                    receivers=total_receivers,
                    senders=total_senders,
                    n_node=total_n_nodes,
                    n_edge=total_n_edges
                )
                xfer_tuples.append(this_xfer_tuple)
            else:
                xfer_tuples.append(None)

            num_locations = min(self.num_locations, len(xfer))
            location_mask = [1] * num_locations + [0] * (self.num_locations - num_locations)
            location_masks.append(location_mask)

        location_masks = np.asarray(location_masks)

        return dict(graph=graph_tuple, xfers=xfer_tuples, location_mask=location_masks, mask=xfer_mask)

    def get_num_actions(self):
        return self.rl_opt.get_num_xfers()

    def step(self, actions):
        pass


class HierarchicalEnvironment(_BaseEnvironment):
    def __init__(self, num_locations=100, real_measurements=False, reward_function=None):
        super().__init__(num_locations, real_measurements, reward_function)

    def step(self, actions):
        xfer_id, location_id = actions

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


class WorldModelEnvironment(_BaseEnvironment):
    def __init__(self, num_locations, real_measurements, reward_function):
        super().__init__(num_locations, real_measurements, reward_function)

    def step(self, actions):
        raise NotImplementedError

