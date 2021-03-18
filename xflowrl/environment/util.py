import numpy as np
import graph_nets as gn
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
        if sender_id not in guid_to_id:
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