import motile
import networkx as nx


def solution2graph(solver, base_graph, label_key="show"):
    """Convert a solver solution to a graph and corresponding dense selected detections.

    Args:
        solver: A solver instance
        base_graph: The base graph
        detections: The detections
        label_key: The key of the label in the detections
    Returns:
        track_graph: Solution as motile.TrackGraph
        graph: Solution as networkx graph
        selected_detections: Dense label array containing only selected detections

    This functions has been taken from https://github.com/dlmbl/tracking
    """
    graph = nx.DiGraph()
    node_indicators = solver.get_variables(motile.variables.NodeSelected)
    edge_indicators = solver.get_variables(motile.variables.EdgeSelected)
    # Build nodes
    for node, index in node_indicators.items():
        if solver.solution[index] > 0.5:
            node_features = base_graph.nodes[node]
            graph.add_node(node, **node_features)
            node_features[base_graph.frame_attribute]

    # Build edges
    for edge, index in edge_indicators.items():
        if solver.solution[index] > 0.5:
            graph.add_edge(*edge, **base_graph.edges[edge])

    # Add cell division markers on edges for traccuracy
    for (u, v), features in graph.edges.items():
        out_edges = graph.out_edges(u)
        if len(out_edges) == 2:
            features["is_intertrack_edge"] = 1
        elif len(out_edges) == 1:
            features["is_intertrack_edge"] = 0
        else:
            raise ValueError()

    track_graph = motile.TrackGraph(graph, frame_attribute="time")

    return track_graph, graph  # selected_detections
