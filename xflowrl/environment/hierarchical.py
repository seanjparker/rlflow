from collections import deque
from functools import partial

import numpy as np
import graph_nets as gn

from xflowrl.core import PyRLOptimizer
from taso.core import op_table

# Build op table
op_tbl = {}
for num, op_str in enumerate(sorted(op_table.values())):
    op_tbl[op_str] = num
op_tbl["Unknown"] = -1


def graph_to_graphnet_tuple(
        graph,
        op_runtime_callback=lambda guid: 0.0,
        max_input_dims=4,
        start_n_node=0,
        start_n_edge=0,
        return_graphstuple=True
):
    globals = np.asarray([[
        graph.cost()
    ]], dtype=np.float32)

    guid_to_id = {}

    nodes = {}
    edges = {}
    receivers = {}
    senders = {}

    current_edge_id = start_n_edge
    for current_node_id, node in enumerate(graph.get_operator_list(), start_n_node):
        node_guid = node['guid']
        guid_to_id[node_guid] = current_node_id

        # node embedding
        try:  # e.g. enlarge op is missing from op table, catch assertion errors
            op_type = op_tbl[graph.get_operator_type(node)]
        except AssertionError:
            op_type = -1

        node_val = [
            op_type,
            op_runtime_callback(node)
        ]
        # node_val = op_runtime_callback(node)

        nodes[current_node_id] = node_val

        # loop through input edges
        for idx, edge in enumerate(graph.get_input_edges(node)):
            sender_node = edge['srcOp']
            sender_id = sender_node['guid']

            # Edge embedding
            input_dims = graph.get_input_dims(node, idx)
            edge_val = [input_dims[i] if i < len(input_dims) else 0 for i in range(max_input_dims)]
            edges[current_edge_id] = edge_val

            senders[current_edge_id] = sender_id  # Attention: This is a guid and has to be re-wired
            receivers[current_edge_id] = current_node_id
            current_edge_id += 1

    # Re-wire senders
    for edge_id, sender_id in senders.items():
        if not sender_id in guid_to_id:
            senders[edge_id] = None
        else:
            senders[edge_id] = guid_to_id[sender_id]

    n_node = [len(nodes)]
    n_edge = [len(edges)]

    nodes = np.asarray([nodes[node_id] for node_id in sorted(nodes.keys())], dtype=np.float32)
    edges = np.asarray([edges[edge_id] for edge_id in sorted(edges.keys())], dtype=np.float32)
    senders = np.asarray([senders[edge_id] or receivers[edge_id] or 0 for edge_id in sorted(senders.keys())], dtype=np.int32)  # Todo: Add a dummy node.
    receivers = np.asarray([receivers[edge_id] for edge_id in sorted(receivers.keys())], dtype=np.int32)

    # Todo: Add a maximum of nodes/edges to allow to train multiple different graphs
    if not return_graphstuple:
        return nodes, edges, globals, receivers, senders, n_node, n_edge

    return gn.graphs.GraphsTuple(
        nodes=nodes,
        edges=edges,
        globals=globals,
        receivers=receivers,
        senders=senders,
        n_node=n_node,
        n_edge=n_edge
    )


class HierarchicalEnvironment(object):
    def __init__(self, num_locations=100, real_measurements=False, reward_function=None):
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

