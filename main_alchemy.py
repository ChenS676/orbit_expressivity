import random
import argparse
import networkx as nx
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT, GCN
import wandb

from losses import OrbitSortingCrossEntropyLoss, CrossEntropyLossWrapper
from models import RniGCN, UniqueIdGCN, UniqueIdDeepSetsGCN, OrbitIndivGCN, MaxOrbitGCN, CustomPygGCN, RniMaxPoolGCN
from testing import model_accuracy
from datasets import MaxOrbitGCNTransform

parser = argparse.ArgumentParser()

# logging options
parser.add_argument('--loss_log_interval', type=int, default=10)
parser.add_argument('--use_wandb', type=int, default=1)

# model
parser.add_argument('--model', type=str, default='max_orbit_gcn',
                    choices=['gcn', 'gat', 'unique_id_gcn', 'rni_gcn', 'orbit_indiv_gcn', 'max_orbit_gcn'])
parser.add_argument('--gnn_layers', type=int, default=4)
parser.add_argument('--gnn_hidden_size', type=int, default=40)
parser.add_argument('--rni_channels', type=int, default=10)
# max orbit of max-orbit model, only used for max_orbit_gcn
parser.add_argument('--model_max_orbit', type=int, default=6)

# dataset
parser.add_argument('--train_on_entire_dataset', type=int, default=1)
# how much of the data should be used for the train set
parser.add_argument('--train_split', type=float, default=0.9)
# filter out non-equivariant examples from the bioisostere dataset
parser.add_argument('--bioisostere_only_equivariant', type=int, default=0)
parser.add_argument('--dataset', type=str, default='alchemy',
                    choices=['bioisostere', 'mutag', 'alchemy', 'zinc'])
# use with alchemy to create a max_orbit dataset
parser.add_argument('--max_orbit_alchemy', type=int, default=6)
# when creating a max_orbit dataset, shuffle the target order within the max orbits
parser.add_argument('--shuffle_targets_in_max_orbit', type=int, default=1)
parser.add_argument('--shuffle_dataset', type=int, default=0)

# training
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--changed_node_loss_weight', type=float, default=1)
parser.add_argument('--loss', type=str, default='orbit_sorting_cross_entropy',
                    choices=['cross_entropy', 'orbit_sorting_cross_entropy'])

# evaluation
parser.add_argument('--train_eval_interval', type=int, default=10)
parser.add_argument('--test_eval_interval', type=int, default=10)

# misc
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use_cpu', type=int, default=0)

args = parser.parse_args()

# init logging
if args.use_wandb:
    name_tag = f"{args.dataset}_{args.model}_{args.model_max_orbit}"
    wandb.init(project="orbit-gnn", name=name_tag, config=vars(args))

# fix RNG
if args.seed == 0:  # sample seed at random
    args.seed = random.randint(1, 10000)
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cpu:
    device = 'cpu'

# G = nx.Graph()
#
# G.add_nodes_from([
#     (0, {'x': (1.0, 1.0), 'y': 1}),
#     (1, {'x': (2.0, 1.0), 'y': 1}),
# ])
#
# for i in range(2, 7):
#     G.add_node(i, **{'x': (1.0, 1.0), 'y': 1})
#
# G.add_edges_from([
#     (0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (4, 6)
# ])

dataset = None
if args.dataset == 'alchemy':
    # alchemy_nx, num_node_classes = nx_molecule_dataset('alchemy_full')
    # if args.max_orbit_alchemy >= 2:
    #     orbit_alchemy_nx = alchemy_max_orbit_dataset(
    #         dataset=alchemy_nx,
    #         num_node_classes=num_node_classes,
    #         extended_dataset_size=100000,  
    #         max_orbit=args.max_orbit_alchemy,
    #         shuffle_targets_within_orbits=args.shuffle_targets_in_max_orbit,
    #     )
    #     orbit_alchemy_pyg = pyg_max_orbit_dataset_from_nx(orbit_alchemy_nx)
    dataset = torch.load(f'{args.dataset}/alchemy_max_orbit_{args.max_orbit_alchemy}.pt')
else:
    raise Exception('Dataset "', args.dataset, '" not recognized')

# set up loss
if args.loss == 'cross_entropy':
    criterion = CrossEntropyLossWrapper()
elif args.loss == 'orbit_sorting_cross_entropy':
    criterion = OrbitSortingCrossEntropyLoss()
else:
    raise Exception('Loss "', args.loss, '" not recognized')

# set number of input and output channels
in_channels = dataset[0].x.size()[1]
out_channels = in_channels  # same number of classes by default
if args.dataset == 'bioisostere':
    out_channels += 1
if args.dataset == 'alchemy' and out_channels < args.max_orbit_alchemy + 1:
    out_channels = args.max_orbit_alchemy + 1


max_orbit_transform = None
if args.model == 'max_orbit_gcn':
    max_orbit_transform = MaxOrbitGCNTransform(args.model_max_orbit, out_channels)
    max_orbit_transform.transform_dataset(dataset)
    out_channels += 1

# possibly shuffle the dataset
if args.shuffle_dataset:
    random.shuffle(dataset)

# set up train / test split on dataset
train_dataset = dataset[0:int(len(dataset) * args.train_split)]
if args.train_on_entire_dataset:
    train_dataset = dataset
test_dataset = dataset[int(len(dataset) * args.train_split):]

print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))

# set up model
if args.model == 'gat':
    model = GAT(
        in_channels=in_channels,
        hidden_channels=args.gnn_hidden_size,
        num_layers=args.gnn_layers,
        out_channels=out_channels,
    )
elif args.model == 'gcn':
    model = CustomPygGCN(
        in_channels=in_channels,
        hidden_channels=args.gnn_hidden_size,
        num_layers=args.gnn_layers,
        out_channels=out_channels,
    )
elif args.model == 'unique_id_gcn':
    model = UniqueIdGCN(
        in_channels=in_channels,
        hidden_channels=args.gnn_hidden_size,
        num_layers=args.gnn_layers,
        out_channels=out_channels
    )
elif args.model == 'rni_gcn':
    model = RniMaxPoolGCN(
        in_channels=in_channels,
        hidden_channels=args.gnn_hidden_size,
        num_layers=args.gnn_layers,
        out_channels=out_channels,
        rni_channels=args.rni_channels,
    )
elif args.model == 'orbit_indiv_gcn':
    model = OrbitIndivGCN(
        in_channels=in_channels,
        hidden_channels=args.gnn_hidden_size,
        num_layers=args.gnn_layers,
        out_channels=out_channels,
    )
elif args.model == 'max_orbit_gcn':
    model = MaxOrbitGCN(
        in_channels=in_channels,
        hidden_channels=args.gnn_hidden_size,
        num_layers=args.gnn_layers,
        out_channels=out_channels,
        max_orbit=args.model_max_orbit,
    )
else:
    raise Exception('Model "', args.model, '" not recognized')

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

# train
print('Training model')
model.train()
for epoch in range(args.n_epochs):
    epoch_loss = 0
    for data in train_dataset:
        optimizer.zero_grad()
        data = data.to(device)  # TODO: optimize code for GPU
        model = model.to(device)
        out = model(data.x, data.edge_index, orbits=data.orbits)
        targets = data.transformed_y if args.model == 'max_orbit_gcn' else data.y
        loss = criterion(out, targets, data.non_equivariant_orbits)

        loss.backward()
        optimizer.step()
        epoch_loss += loss

    # log results from epoch
    if (epoch + 1) % args.loss_log_interval == 0:
        # log the loss
        print('Epoch:', epoch + 1, '| Epoch loss:', epoch_loss.item())

        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
            }, step=epoch + 1)

    if (epoch + 1) % args.train_eval_interval == 0:
        # compute train accuracy
        model.training = False
        node_accuracy, orbit_accuracy, graph_accuracy = model_accuracy(
            train_dataset, model, device, max_orbit_transform)
        model.training = True
        print('Epoch:', epoch + 1, '| Eval on training dataset | Node accuracy:', node_accuracy,
              '| Orbit accuracy:', orbit_accuracy, '| Graph accuracy:', graph_accuracy)

        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_node_accuracy': node_accuracy,
                'train_orbit_accuracy': orbit_accuracy,
                'train_graph_accuracy': graph_accuracy,
            }, step=epoch + 1)

    if (epoch + 1) % args.test_eval_interval == 0:
        # compute test accuracy
        model.training = False
        node_accuracy, orbit_accuracy, graph_accuracy = model_accuracy(
            test_dataset, model, device, max_orbit_transform)
        model.training = True
        print('Epoch:', epoch + 1, '| Eval on test dataset | Node accuracy:', node_accuracy,
              '| Orbit accuracy:', orbit_accuracy, '| Graph accuracy:', graph_accuracy)

        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'test_node_accuracy': node_accuracy,
                'test_orbit_accuracy': orbit_accuracy,
                'test_graph_accuracy': graph_accuracy,
            }, step=epoch + 1)

