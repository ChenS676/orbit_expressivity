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


for depth in range(2, 6):  
        branch = 2  
        name = 'balenced_tree'
        tree = nx.balanced_tree(r=branch, h=depth)
        pos = nx.spring_layout(tree, seed=42)
        node_mapping = {node: idx for idx, node in enumerate(tree.nodes())}
        adj_dict = {
                node_mapping[node]: [node_mapping[neighbor] for neighbor in tree.neighbors(node)]
                for node in tree.nodes()
        }

        n = len(tree.nodes())
        G_pynauty = pynauty.Graph(number_of_vertices=n, adjacency_dict=adj_dict, directed=False)
        generators, _, _, orbits, num_orbit = pynauty.autgrp(G_pynauty)
        new_label_mapping = {node: idx for idx, node in enumerate(set(orbits))}
        orbits = [new_label_mapping[orbit] for orbit in orbits]

        print(f"Number of orbits: {num_orbit}, Tree depth: {depth}")
        node_colors = [orbits[node] for node in range(n)]

        custom_labels = {}
        for orbit_idx, orbit in enumerate(orbits):
                custom_labels[orbit_idx] = f"{orbits[orbit_idx]}"  # Label all nodes in the orbit with its index

        plt.figure(figsize=(8, 6))
        nx.draw(
        tree,
        pos,
        with_labels=True,
        labels=custom_labels,
        node_color=node_colors,
        cmap='Blues',
        node_size=500,
        edge_color="gray",
        edgecolors='black'
        )
        plt.title(f"{name.capitalize()} Tree Orbits (depth={depth})")
        plt.savefig(f"{name}_tree_orbits_d{depth}.pdf")
        plt.close()


        plt.figure(figsize=(6, 4))
        plt.hist(orbits, bins=range(min(orbits), max(orbits) + 2), align='left', rwidth=0.8)
        plt.xlabel('Orbit Label')
        plt.ylabel('Frequency')
        plt.title('Histogram of Orbit Labels')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        plt.savefig(f"{name}_distribution_d{depth}.pdf")
        plt.close()



for N in range(4, 9):
        name = 'grid'
        G_nx = nx.grid_2d_graph(N, N)
        pos_original = {node: (node[1], -node[0]) for node in G_nx.nodes()}  
        node_mapping = {node: idx for idx, node in enumerate(G_nx.nodes())}
        adj_dict = {node_mapping[node]: [node_mapping[neighbor] for neighbor in G_nx.neighbors(node)] 
                for node in G_nx.nodes()}
        n = len(G_nx.nodes())
        G_pynauty = pynauty.Graph(number_of_vertices=n, adjacency_dict=adj_dict, directed=False)
        

        # return -> (generators, grpsize1, grpsize2, orbits, numorbits)
        #     For the detailed description of the returned components, see
        generators, _, _,  orbits, num_orbit  = pynauty.autgrp(G_pynauty)
        new_label_mapping = {node: idx for idx, node in enumerate(set(orbits))}
        orbits = [new_label_mapping[orbit] for orbit in orbits]
        custom_labels = {}
        for i, ov in zip(G_nx.nodes(), orbits):
                custom_labels[i] = f"{ov}"

        plt.figure(figsize=(8, 6))
        node_colors = [orbits[node] for node in range(n)]  
        nx.draw(G_nx,  
                pos=pos_original, 
                # with_labels=True, 
                labels=custom_labels, 
                node_color=node_colors, 
                cmap='Blues',
                node_size=500, 
                font_weight='bold',
                edgecolors='black')
        plt.savefig(f"orbits{N}_{name}new.pdf")
        plt.close()
        
        plt.figure(figsize=(6, 4))
        plt.hist(orbits, 
                 bins=range(min(orbits), max(orbits) + 2), 
                 align='left', 
                 rwidth=0.8)
        plt.xlabel('Orbit Label')
        plt.ylabel('Frequency')
        plt.title('Histogram of Orbit Labels')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        plt.savefig(f"{name}_distribution_d{depth}_new.pdf")
        plt.close()

