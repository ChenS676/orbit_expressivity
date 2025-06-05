import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_labeled_graph(graph: nx.Graph, orbits: Optional[List[List[int]]] = None, show_node_id: bool = True, index: int = 0) -> None:
    pos = nx.spring_layout(graph, seed=1)
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 1}

    node_color = {}
    for orbit_idx, orbit in enumerate(orbits):
            node_color[orbit_idx] = f"{orbits[orbit_idx]}"  # Label all nodes in the orbit with its index

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(graph, 
                           pos, 
                           **options, 
                           cmap='Blues')
    nx.draw_networkx_edges(graph, pos, width=1, alpha=0.5)
    labels = nx.get_node_attributes(graph, 'x')

    if show_node_id:
        for node, label in labels.items():
            labels[node] = str(node) + ':' + str(label)
    nx.draw_networkx_labels(graph, pos, labels=node_color, font_size=10, font_color='black')
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(f'example_{index}.pdf', format='pdf')
    plt.close()
