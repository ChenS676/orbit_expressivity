from typing import Tuple, List, Optional
import networkx as nx
from torch_geometric.datasets import TUDataset
import matplotlib.pyplot as plt


def multi_set_hash_function(input_set: list) -> int:
    return hash(tuple(sorted(input_set)))


def wl_gc(graph: nx.Graph) -> Tuple[int, List[int]]:
    """wl test for graph classification"""
    labels = [-1] * len(graph)
    node_attributes = nx.get_node_attributes(graph, 'x')
    if node_attributes:
        for node in graph.nodes:
            labels[node] = hash(tuple(node_attributes[node]))
    num_unique = len(set(labels))

    for wl_iteration in range(1, len(graph) + 1):
        previous_labels = labels[:]
        previous_num_unique = num_unique
        global_hash = multi_set_hash_function(previous_labels)
        # global hash is the hash of all vertices hashes
        for node in graph.nodes:
            neighbours = graph[node]
            neighbour_labels = [previous_labels[neighbour] for neighbour in neighbours]
            neighbour_hash = multi_set_hash_function(neighbour_labels)
            combined_hash = hash((previous_labels[node], neighbour_hash, global_hash))
            labels[node] = combined_hash

        num_unique = len(set(labels))
        if num_unique == previous_num_unique:
            return wl_iteration, labels
    raise Exception('WL did not converge: something is wrong with the algorithm')


def compute_wl_hash(graph: nx.Graph) -> int:
    _, final_labels = wl_gc(graph)
    return multi_set_hash_function(final_labels)


def compute_wl_orbits(graph: nx.Graph) -> Tuple[int, List[List[int]]]:
    n_iterations, final_labels = wl_gc(graph)
    node_list = list(graph.nodes)[1:]
    orbits = [[list(graph.nodes)[0]]]

    for node in node_list:
        found_orbit = False
        for orbit_index, orbit in enumerate(orbits):
            orbit_node = orbit[0]
            if final_labels[node] == final_labels[orbit_node]:
                orbits[orbit_index].append(node)
                found_orbit = True
                break
        if not found_orbit:
            orbits.append([node])

    return n_iterations, orbits


def plot_labeled_graph(graph: nx.Graph, orbits: Optional[List[List[int]]] = None, show_node_id: bool = True):
    pos = nx.spring_layout(graph, seed=1)
    options = {"edgecolors": "tab:gray", "node_size": 100, "alpha": 1}

    node_color = [0] * len(graph)
    if orbits is not None:
        for node in graph.nodes:
            orbit_index = 0
            for i, orbit in enumerate(orbits):
                if node in orbit:
                    orbit_index = i
                    break
            node_color[node] = orbit_index + 1 if len(orbits[orbit_index]) > 1 else 0
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(graph, pos, **options, node_color=node_color, cmap='Set1')
    nx.draw_networkx_edges(graph, pos, width=1, alpha=0.5)
    labels = nx.get_node_attributes(graph, 'x')

    if show_node_id:
        for node, label in labels.items():
            labels[node] = f"{node}: {tuple(label)}"

    nx.draw_networkx_labels(graph, pos, labels, font_size=10, font_color='black')
    plt.tight_layout()
    plt.axis("off")
    plt.savefig("graph_plot.pdf", dpi=300)


def to_networkx(graph) -> nx.Graph:
    G = nx.Graph()
    edge_index = graph.edge_index.numpy()
    x = graph.x.numpy()
    y = graph.y.item()

    for i in range(x.shape[0]):
        G.add_node(i, x=x[i], y=y)

    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        G.add_edge(u, v)

    return G


def generate_grid_graph(m: int, n: int) -> nx.Graph:
    """
    Generate a 2D grid graph with m rows and n columns.
    Nodes will have attribute 'x' set to their (row, col) coordinates.
    Nodes will be relabeled to integers 0,...,m*n-1.
    """
    G_2d = nx.grid_2d_graph(m, n)
    G = nx.convert_node_labels_to_integers(G_2d, ordering='sorted')
    for node in G.nodes:
        # original pos in 2D grid as attribute
        orig_pos = list(G_2d.nodes)[node]
        G.nodes[node]['x'] = orig_pos
    return G




def generate_triangular_grid(m: int, n: int) -> nx.Graph:
    """
    Generate a triangular grid graph with m rows and n columns of triangles.

    Each node is placed on a 2D lattice with edges forming triangles.
    """
    G = nx.Graph()
    # Create nodes
    for i in range(m + 1):
        for j in range(n + 1):
            G.add_node(i * (n + 1) + j, x=(i, j))  # store position as attribute for hashing

    # Add edges forming triangles
    for i in range(m):
        for j in range(n):
            # Nodes in the current cell square
            top_left = i * (n + 1) + j
            top_right = top_left + 1
            bottom_left = (i + 1) * (n + 1) + j
            bottom_right = bottom_left + 1

            # Add edges of the square
            G.add_edge(top_left, top_right)
            G.add_edge(top_left, bottom_left)
            G.add_edge(top_right, bottom_right)
            G.add_edge(bottom_left, bottom_right)

            # Add the diagonal to form two triangles
            if (i + j) % 2 == 0:
                # diagonal from top_left to bottom_right
                G.add_edge(top_left, bottom_right)
            else:
                # diagonal from top_right to bottom_left
                G.add_edge(top_right, bottom_left)

    # Assign node attribute 'x' as a tuple of the node's position for hashing
    for node in G.nodes:
        G.nodes[node]['x'] = G.nodes[node]['x']

    return G

def main():
    # MUTAG test
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    graph = dataset[0]
    nx_graph = to_networkx(graph)

    # wl_hash = compute_wl_hash(nx_graph)
    iterations, orbits = compute_wl_orbits(nx_graph)


    print("MUTAG graph test:")
    # print(f"WL Hash: {wl_hash}")
    print(f"Converged in {iterations} iterations.")
    print("Orbits:", orbits)
    plot_labeled_graph(nx_graph, orbits=orbits, show_node_id=False)
    import IPython; IPython.embed()  # For debugging purposes
    
    # Grid graph test
    grid_graph = generate_grid_graph(4, 4)  # 4x4 grid
    wl_hash_grid = compute_wl_hash(grid_graph)
    iterations_grid, orbits_grid = compute_wl_orbits(grid_graph)

    print("\nGrid graph test:")
    print(f"WL Hash: {wl_hash_grid}")
    print(f"Converged in {iterations_grid} iterations.")
    print("Orbits:", orbits_grid)
    plot_labeled_graph(grid_graph, orbits=orbits_grid, show_node_id=False)
    import IPython; IPython.embed()  # For debugging purposes


if __name__ == "__main__":
    main()