import networkx as nx
import pynauty
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib.pyplot as plt

import os
import glob

# Specify the folder path
folder_path = '/hkfs/work/workspace/scratch/cc7738-automorphism/orbit-gnn'

# Use glob to get all the PDF files in the folder
pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))

# Loop through all the files and delete them
for pdf_file in pdf_files:
    try:
        os.remove(pdf_file)
        print(f"Deleted: {pdf_file}")
    except Exception as e:
        print(f"Error deleting {pdf_file}: {e}")



from graph_theory import compute_orbits
from plotting import plot_labeled_graph
from torch_geometric.datasets import TUDataset
import torch
from torch_geometric.utils import to_networkx


import networkx as nx
import pynauty

def automorphism(graph: nx.Graph, seed=42):
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    
    adj_dict = {
        node_mapping[node]: [node_mapping[neighbor] for neighbor in graph.neighbors(node)]
        for node in graph.nodes()
    }
    n = len(graph.nodes())
    G_pynauty = pynauty.Graph(number_of_vertices=n, 
                              adjacency_dict=adj_dict, 
                              directed=False)
    _, _, _, orbits, num_orbit = pynauty.autgrp(G_pynauty)
    new_label_mapping = {node: idx for idx, node in enumerate(set(orbits))}
    orbits = [new_label_mapping[orbit] for orbit in orbits]
    
    return orbits, num_orbit


def main():
        # Load the MUTAG dataset
        dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
        N = 4
        graph = nx.grid_2d_graph(N, N)
        orbits, num_orbits = automorphism(graph)

        for i in range(len(dataset)):

            graph = dataset[i]
            nx_graph = to_networkx(graph)
            orbits, num_orbits = automorphism(nx_graph)
            n = len(nx_graph.nodes())
            node_colors = [orbits[node] for node in range(n)]
            print("Orbits:", orbits, "Number of orbits:", num_orbits)
            custom_labels = {}
            for orbit_idx, orbit in enumerate(orbits):
                    custom_labels[orbit_idx] = f"{orbits[orbit_idx]}"  # Label all nodes in the orbit with its index
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(8, 6))
            # use different color for each orbit
            nx.draw(
                nx_graph,
                pos,
                with_labels=True,
                labels=custom_labels,
                node_color=node_colors,
                cmap='tab20b',
                node_size=500,
                edge_color="gray",
                edgecolors='black'
            )
            plt.savefig(f"mutag_orbits_auto_{i}.pdf")
            plt.close()

if __name__ == "__main__":
    main()


