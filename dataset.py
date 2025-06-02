import dgl
import os
import torch
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs
from utils import anchor_ego_net, adaptive_anchor_ego_net, dgl2list
import networkx as nx
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle
import sys
sys.path.append("./compute")
import filter as dualsim_filter


class IsoDataset(DGLDataset):
    def __init__(self, raw_dir=None, save_dir=None, force_reload=False, verbose=True, ego=False, adaptive_id=False):
        self.ego = ego
        self.adaptive_id = adaptive_id
        suffix = '.bin'
        self.queryname = 'querys'
        self.targetname = 'targets'
        if self.ego:
            self.queryname += '_ego'
            self.targetname += '_ego'
            if self.adaptive_id:
                self.queryname += '_adapt'
                self.targetname += '_adapt'
        self.queryname += suffix
        self.targetname += suffix
        super(IsoDataset, self).__init__(name='isomorphism',
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    def has_cache(self):
        datasetnames = os.listdir(self.raw_dir)
        if self.queryname in datasetnames and self.targetname in datasetnames:
            return True
        else:
            return False

    def process(self):
        print('process')
        querys_filename = os.path.join(self.raw_dir, "querys.bin")
        targets_filename = os.path.join(self.raw_dir, "targets.bin")
        self.querys, self.targets, self.labels = self._load_graph(querys_filename, targets_filename)
        for idx in range(len(self.querys)):
            query = self.querys[idx]
            pretarget = self.targets[idx]
            label = self.labels[idx].view(1)
            sim = dualsim_filter.dualsim_filter_qt(dgl2list(query), dgl2list(pretarget))
            candnodes = set()
            for qid, gids in sim.items():
                candnodes.update(gids.tolist())
            candnodes.add(torch.nonzero(pretarget.ndata["anchored"] == 1).item())
            candnodes = list(candnodes)
            target = dgl.node_subgraph(pretarget, candnodes)
        if self.ego:
            new_querys = []
            new_targets = []
            new_labels = []
            if self.adaptive_id:
                for idx in range(len(self.querys)): 
                    query = self.querys[idx]
                    target = self.targets[idx]
                    label = self.labels[idx].view(1)
                    qanchor = torch.nonzero(query.ndata["anchored"] == 1).item()
                    tanchor = torch.nonzero(target.ndata["anchored"] == 1).item()
                    q, t = adaptive_anchor_ego_net(query, qanchor, target, tanchor)
                    new_querys.append(q)
                    new_targets.append(t)
                    new_labels.append(label)
            else:
                for idx in range(len(self.querys)):
                    query = self.querys[idx]
                    target = self.targets[idx]
                    label = self.labels[idx].view(1)
                    qanchor = torch.nonzero(query.ndata["anchored"] == 1).item()
                    tanchor = torch.nonzero(target.ndata["anchored"] == 1).item()
                    q = anchor_ego_net(query, qanchor)
                    t = anchor_ego_net(target, tanchor)
                    new_querys.append(q)
                    new_targets.append(t)
                    new_labels.append(label)
            self.querys = new_querys
            self.targets = new_targets
            self.labels = torch.cat(new_labels)

    def save(self):
        querys_filename = os.path.join(self.raw_dir, self.queryname)
        targets_filename = os.path.join(self.raw_dir, self.targetname)
        save_graphs(querys_filename, self.querys)
        save_graphs(targets_filename, self.targets, {"label": self.labels})

    def load(self):
        print('load')
        querys_filename = os.path.join(self.raw_dir, self.queryname)
        targets_filename = os.path.join(self.raw_dir, self.targetname)
        self.querys, acnum = load_graphs(querys_filename)
        self.targets, labels_dict = load_graphs(targets_filename)
        self.labels = labels_dict["label"]

    def __getitem__(self, idx):
        return self.querys[idx], self.targets[idx], self.labels[idx]

    def __len__(self):
        return len(self.targets)

    def _load_graph(self, querys_filename, targets_filename):
        querys, _ = load_graphs(querys_filename)
        targets, labels_dict = load_graphs(targets_filename)
        labels = labels_dict["label"]
        return querys, targets, labels
