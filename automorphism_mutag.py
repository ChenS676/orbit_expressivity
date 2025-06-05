import networkx as nx
import pynauty
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib.pyplot as plt

import os
import glob

from graph_theory import compute_orbits
from plotting import plot_labeled_graph
from torch_geometric.datasets import TUDataset
import torch
from torch_geometric.utils import to_networkx
from typing import Tuple, List, Optional
import networkx as nx
from torch_geometric.datasets import TUDataset
import matplotlib.pyplot as plt
from datasets import nx_molecule_dataset, pyg_max_orbit_dataset_from_nx, alchemy_max_orbit_dataset
import pynauty

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


def mutag():
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


def main():
    for max_orbit_alchemy in [7, 8, 9, 10]:
        shuffle_targets_in_max_orbit = 1
        alchemy_nx, num_node_classes = nx_molecule_dataset('alchemy_full')
        if max_orbit_alchemy >= 2:
            orbit_alchemy_nx = alchemy_max_orbit_dataset(
                dataset=alchemy_nx,
                num_node_classes=num_node_classes,
                extended_dataset_size=1000,  # TODO: make arg
                max_orbit=max_orbit_alchemy,
                shuffle_targets_within_orbits=shuffle_targets_in_max_orbit,
            )
            orbit_alchemy_pyg = pyg_max_orbit_dataset_from_nx(orbit_alchemy_nx)
            dataset = orbit_alchemy_pyg
            torch.save(dataset, f'alchemy_max_orbit_{max_orbit_alchemy}.pt')
if __name__ == "__main__":
    main()


