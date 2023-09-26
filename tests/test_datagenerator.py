"""
Test module to test the graph data generator
"""

import networkx as nx

from deepgraphgen.datageneration import generate_dataset, generated_graph, bfs_order


def test_generate_one_graph():
    # test
    graph = generated_graph("erdos_renyi_graph", n=500, p=0.01)

    #
    graph = generated_graph("watts_strogatz_graph", n=500, k=5, p=0.01)

    #
    graph = generated_graph("barabasi_albert_graph", n=500, m=5)

    #
    graph = generated_graph("random_lobster", n=500, p1=0.01, p2=0.01)


def test_bfs_reordering():
    graph = generated_graph("erdos_renyi_graph", n=10, p=0.2)

    graph_reorder = bfs_order(graph)

    graph = generated_graph("grid_graph", nx=10, ny=10)

    graph_reorder = bfs_order(graph)

    edges = list(graph_reorder.edges)
