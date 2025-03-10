"""
python module used to generate graph data (used to trained the different model)

From the networkx documentation we will use 4 graph generators :
- erdos_renyi_graph(n, p[, seed, directed])
- watts_strogatz_graph(n, k, p[, seed])
- barabasi_albert_graph(n, m[, seed, ...])
- random_lobster(n, p1, p2[, seed])

"""
from networkx import (
    erdos_renyi_graph,
    watts_strogatz_graph,
    barabasi_albert_graph,
    random_lobster,
    grid_graph,
    bfs_edges,
    relabel_nodes,
    convert_node_labels_to_integers,
    random_labeled_tree,
)


def generated_graph(
    graph_name, n=None, p=None, k=None, m=None, p1=None, p2=None, nx=None, ny=None
):
    """
    Functions used to generate different graphs
    """
    if graph_name == "erdos_renyi_graph":
        assert p != None, "define p"
        return erdos_renyi_graph(n, p)
    elif graph_name == "watts_strogatz_graph":
        assert p != None, "define p"
        return watts_strogatz_graph(n, k, p)
    elif graph_name == "barabasi_albert_graph":
        assert m != None, "define m"
        return barabasi_albert_graph(n, m)
    elif graph_name == "random_lobster":
        assert p1 != None and p2 != None, "define p1 and p2"
        return random_lobster(n, p1, p2)
    elif graph_name == "grid_graph":
        assert nx != None and ny != None, "define nx and ny"
        return grid_graph(dim=[nx, ny])
    elif graph_name == "random_labeled_tree":
        return random_labeled_tree(n)


def bfs_order(graph):
    """
    Function used to reorder the graph with BFS (starting from node 0)
    """

    graph = convert_node_labels_to_integers(graph)

    list_edges_visited = list(bfs_edges(graph, 0))

    nodes_ordering = [0] + [edge[1] for edge in list_edges_visited]
    mapping = {node: nodes_ordering.index(node) for node in nodes_ordering}

    H = relabel_nodes(graph, mapping)

    return H


def generate_dataset(
    graph_name,
    nb_graphs,
    n=None,
    p=None,
    k=None,
    m=None,
    p1=None,
    p2=None,
    nx=None,
    ny=None,
    bfs_order_activation=True,
):
    """
    Function used to generate a dataset of graphs
    """
    list_graphs = []

    for _ in range(nb_graphs):
        list_graphs.append(generated_graph(graph_name, n, p, k, m, p1, p2, nx, ny))

    # reorder graph with BFS (starting from node 0)
    if bfs_order_activation:
        list_graphs = [bfs_order(graph) for graph in list_graphs if graph.number_of_nodes() > 1]

    return list_graphs
