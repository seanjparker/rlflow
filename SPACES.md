# Spaces

We modeled the state and action space and the reward function in accordance to
prior research. As explained before, we use an iterative approach where at each
step, a decision to apply a transfer to a specific location is made. Here we
will outline the state, action, and reward spaces.

## State space

The state coming from the `HierarchicalEnvironment` consists of a 4-value Tuple:
`(graph_tuple, xfer_tuples, location_masks, xfer_mask)`

### Graph Tuple

The current state of the computation graph we are optimising is converted into
a `gn.graphs.GraphsTuple` object. This is done in the function 
`graph_to_graphnet_tuple()` function. The node attributes is a tuple of the
node id and the op runtime:

```python
node_val = [
    op_type,
    op_runtime_callback(node)
]
```

The edge attributes are the size of the node input dimensions. Per default,
only up to 4 input dimension sizes are reported (which is sensible for image data:
one batch dimension, two size dimensions, one channel dimension).

```python
edge_val = [input_dims[i] if i < len(input_dims) else 0 for i in range(max_input_dims)]
```

These choices were inspired by previous research. The input/output sizes are
exposed by [Grappler](https://openreview.net/pdf?id=Hkc-TeZ0W) and 
[PlaceTo](https://arxiv.org/pdf/1906.08879.pdf). Grappler also exposes an operation
ID (`op_type` in our case). PlaceTo exposes information on the placement and 
decision state, but since we are not optimising placement and have a dynamic
graph it does not make sense to include this information (it is not well-defined).
From a tensor-graph perspective, this should cover all relevant parameters,
but there might be some problem-specific parameters we are currently missing.

### XFER tuples

Each possible transformation would lead to an altered graph. These graphs are
already stored by the TASO backend to calculate the estimated run time. Thus,
we expose the total set of candidate output graphs to the state.

Each candidate graph is passed through the same `graph_to_graphnet_tuple()`
function as the current graph (see above). The object returned in the state 
is the multi-graph `GraphTuple`.

### XFER masks

There is a set number of possible transformations (currently 151). Not every
transformation can be applied to every graph. If the current graph e.g. only
has 4 possible transformations that can be applied, all other XFER IDs
are invalid. We thus return a boolean location mask where only valid XFER IDs
are set to 1. This can be used to zero-out the logits of invalid XFERs (and
thus actions) to make sure the agent always selects a valid XFER.

This is a `N`-dimensional array, where *N* is the number of XFERs
(currently 151).


### Location masks

In the same vein, for each transformation selected by the agent, there are
a number of valid locations where this transformation can be applied.
We limit the number of locations to 100 (can be configured).
If the current graph has fewer than 100 possible for a given XFER, 
e.g. 60 possible locations, the remaining locations (e.g. 40) are
invalid actions. Thus we return a boolean location mask which can be used to
zero out the logits of invalid locations. We do this for every XFER, so this
is a `N x L` array, where *N* is the number of XFERs (currently 151) and `L`
is the maximum number of valid locations (per default 100).

## Action space

The environment expects two actions: One to select the XFER to apply, and
one to select the location where to apply the XFER at.

The action is thus a 2-value tuple: `(xfer_id, location)`. 

There is a special case for the `xfer_id`. If this is `N` (i.e. 151), this
is the *no-op* action. This will not  apply any XFER to the graph, but instead
end the current episode (the `done` value will be True and depending on the
reward mechanism, the reward will be calculated).

Please note here that the IDs are 0-indexed, so `xfer_id=151` would mean
the 152nd XFER, which currently does not exist.

## Reward space

In the current implementation, the reward at each step is the graph runtime
of the previous step minus the altered graph runtime, i.e. the incremental
improvement of the runtime after the current step:

`Reward(t) = Runtime(t-1) - Runtime(t)`

In an alternative formulation, we could return a 0 reward at each step but the
last. At the last step, the difference between the initial runtime and the
altered graph runtime would be returned:

```plain
Reward(t) = 0 for t < |T|
Reward(t) = Runtime(0) - Runtime(t) for t = |T|
```

To use the second variant, two lines have to be commented out in the environment
reward function (there are comments to point out the specific location).

Note that the former reward function seems to work fine and has been used in
most sequential RL applications.