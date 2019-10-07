from xflowrl.environment.hierarchical import HierarchicalEnvironment


def graph_complexity(graph, name='Untitled', env=None):
    if not env:
        env = HierarchicalEnvironment()
    env.set_graph(graph)
    env.reset()  # Need to do this to get the number of actions1

    locations = env.rl_opt.get_available_locations()

    map = {}

    print("-"*40)
    print("Graph: {}".format(name))
    total_subst = total_loc = 0
    for loc, count in enumerate(locations):
        if count == 0:
            continue
        map[loc] = count
        print("XFER ID: {} - Count: {}".format(loc, count))
        total_subst += 1
        total_loc += count
    print("Total number of available substitutions: {}".format(total_subst))
    print("Total available locations for graph substitutions: {}".format(total_loc))

    return total_subst, total_loc, map
