import os
import glob
import sys
sys.path.append('/hkfs/work/workspace/scratch/cc7738-automorphism/orbit-gnn/')  # Replace with the actual path


import networkx as nx
import pynauty
import matplotlib.pyplot as plt
import numpy as np
from graph_theory import compute_orbits
from plotting import plot_labeled_graph
from torch_geometric.datasets import TUDataset
import torch
from torch_geometric.utils import to_networkx
from typing import Tuple, List, Optional
import networkx as nx
from torch_geometric.datasets import TUDataset
from datasets import nx_molecule_dataset, pyg_max_orbit_dataset_from_nx, alchemy_max_orbit_dataset
import pynauty
from collections import Counter


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



alchemy_max_orbit_2 = torch.load('/hkfs/work/workspace/scratch/cc7738-automorphism/orbit-gnn/alchemy_max_orbit_unique_dataset.pt', map_location='cpu')
dataset = alchemy_max_orbit_2 
max_orbit_all = 0
max_orbit_size_list = []
for i in range(len(dataset)):
    
    graph = dataset[i]
    #nx_graph = to_networkx(graph)
    orbits, num_orbits = automorphism(graph)
    orbit_counts = Counter(orbits)
    
    max_orbit_size = max(orbit_counts.values())
    max_orbit_size_list.append(max_orbit_size)
    if max_orbit_size > max_orbit_all:
        max_orbit_all = max_orbit_size
    n = len(graph.nodes())
    node_colors = [orbits[node] for node in range(n)]
    print(
          "Number of orbits:", num_orbits, 
          "max_orbit_size:", max_orbit_size, 
          "max_orbit_all:", max_orbit_all,)
    custom_labels = {}
    for orbit_idx, orbit in enumerate(orbits):
            custom_labels[orbit_idx] = f"{orbits[orbit_idx]}"  # Label all nodes in the orbit with its index
    pos = nx.spring_layout(graph, seed=42)
    
    if max_orbit_size >= 10:
        plt.figure(figsize=(8, 6))
        # use different color for each orbit
        
        nx.draw(
            graph,
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
        plt.show()
        plt.close()

max_orbit_size_list = np.array(max_orbit_size_list)
# # Generate histogram
# plt.figure(figsize=(8, 5))
# plt.hist(max_orbit_size_list, bins=np.arange(min(max_orbit_size_list), max(max_orbit_size_list) + 2) - 0.5, edgecolor='black')
# plt.title('Distribution of max_orbit_size_list')
# plt.xlabel('Orbit Size')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.savefig('max_orbit_dist_3.pdf')
# plt.show()