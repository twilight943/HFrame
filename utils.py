import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import re
import os
import json
import copy
import subprocess
from torch.optim.lr_scheduler import LambdaLR
from collections import OrderedDict
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import random
import networkx as nx
import multiprocessing as mp
import dgl

def adaptive_anchor_ego_net(query, qanchor, target, tanchor, radius=3):
    eattrs = None
    nxquery = dgl.to_networkx(query, node_attrs=['label'], edge_attrs=eattrs)
    Q = nx.ego_graph(nxquery, qanchor, undirected=True, radius=radius)
    nx.set_node_attributes(Q, torch.tensor(0), "id")
    nx.set_node_attributes(Q, torch.tensor(0), "anchored")
    Q.nodes[qanchor]['id'] = torch.tensor(1)
    Q.nodes[qanchor]['anchored'] = torch.tensor(1)
    nxtarget = dgl.to_networkx(target, node_attrs=['label'], edge_attrs=eattrs)
    T = nx.ego_graph(nxtarget, tanchor, undirected=True, radius=radius)
    nx.set_node_attributes(T, torch.tensor(0), "id")
    nx.set_node_attributes(T, torch.tensor(0), "anchored")
    T.nodes[tanchor]['id'] = torch.tensor(1)
    T.nodes[tanchor]['anchored'] = torch.tensor(1)
    query_id = identity_feature(query, 3, [qanchor]).view(-1).tolist()
    target_id = identity_feature(target, 3, [tanchor]).view(-1).tolist()
    assert len(query_id) == len(target_id)
    for qid, tid in zip(query_id, target_id):
        if qid > 0 and tid == 0:
            T.nodes[tanchor]['id'] = torch.tensor(0)
    dglQ = dgl.from_networkx(Q, node_attrs=['label', 'anchored', 'id'], edge_attrs=eattrs)
    dglT = dgl.from_networkx(T, node_attrs=['label', 'anchored', 'id'], edge_attrs=eattrs)
    return dglQ, dglT

def anchor_ego_net(graph, anchor, radius=3):
    eattrs = None
    nxgraph = dgl.to_networkx(graph, node_attrs=['label'], edge_attrs=eattrs)
    G = nx.ego_graph(nxgraph, anchor, undirected=True, radius=radius)
    nx.set_node_attributes(G, torch.tensor(0), "id")
    nx.set_node_attributes(G, torch.tensor(0), "anchored")
    G.nodes[anchor]['id'] = torch.tensor(1)
    G.nodes[anchor]['anchored'] = torch.tensor(1)
    dglG = dgl.from_networkx(G, node_attrs=['label', 'anchored', 'id'], edge_attrs=eattrs)
    return dglG

class BinaryEncoding(nn.Module):
    def __init__(self, max_label):
         super(BinaryEncoding, self).__init__()
         self.out_len = len(bin(max_label)[2:])
    
    def reset_parameters(self):
        pass

    def forward(self, labels):
        encoded_labels = []
        for label in labels:
            binary_str = bin(int(label.item()))[2:]
            padded_binary_str = binary_str.zfill(self.out_len)
            encoded_labels.append([int(bit) for bit in padded_binary_str])
        encoded_tensor = torch.tensor(encoded_labels, dtype=torch.float32).to(labels.device)
        return encoded_tensor

def identity_feature(g: dgl.DGLGraph, length, nodelist):
    graph = g.cpu()
    id_feat_list = []
    graph.add_self_loop()
    if nodelist:
        for n in nodelist:
            id_feat = torch.zeros(length)
            now_successors = [n]
            for i in range(0, length):
                new_successors = []
                for suc in now_successors:
                    new_suc = graph.successors(suc).tolist()
                    new_successors.extend(new_suc)
                id_feat[i] = new_successors.count(n)
                now_successors = [x for x in new_successors if x != n]
            id_feat_list.append(id_feat)
        id_feat_ten = torch.stack(id_feat_list, dim=0)
    else:
        adj = graph.adj().to_dense()
        current_adj = adj
        id_feat = torch.diag(current_adj)
        id_feat_list.append(id_feat)
        for i in range(2, length + 1):
            current_adj = current_adj @ adj
            id_feat = torch.diag(current_adj)
            id_feat_list.append(id_feat)
        id_feat_ten = torch.stack(id_feat_list, dim=-1)
    return id_feat_ten.to(g.device)

def dgl2list(g):
    vid = g.nodes().tolist()
    vlbl = g.ndata['label'].tolist()
    src, dst, eid = g.edges('all')
    srcid = src.tolist()
    dstid = dst.tolist()
    elbl = g.edata['label'].tolist()
    eid = eid.tolist()
    return [vid, vlbl, srcid, dstid, elbl, eid]

def get_norm(norm_type, dim):
    if norm_type == 'batch':
        return nn.BatchNorm1d(dim)
    elif norm_type == 'layer':
        return nn.LayerNorm(dim)
    else:
        return NotImplementedError
    
def get_act(act_type):
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'silu':
        return nn.SiLU()
    elif act_type == 'leaky':
        return nn.LeakyReLU(0.1)
    else:
        return NotImplementedError


def e_value(p_emb, g_emb):
    return torch.sum((torch.clamp(p_emb-g_emb, min=0)) ** 2, dim=1)

def margin_loss(margin, p_emb, g_emb, label, weight=1):
    e = e_value(p_emb, g_emb) 
    e[label == 0] = (torch.clamp(margin-e, min=0))[label == 0]
    if weight != 1:
        e[label == 1] = e[label == 1] * weight
    loss = torch.sum(e)
    return loss


##########################################################
######### Representation and Encoding Functions ##########
##########################################################
def get_enc_len(x, base=10):
    # return math.floor(math.log(x, base)+1.0)
    l = 0
    while x:
        l += 1
        x = x // base
    return l

def int2onehot(x, len_x, base=10):
    if isinstance(x, (int, list)):
        x = np.array(x)
    x_shape = x.shape
    x = x.reshape(-1)
    one_hot = np.zeros((len_x*base, x.shape[0]), dtype=np.float32)
    x =  x % (base**len_x)
    idx = one_hot.shape[0] - base
    while np.any(x):
        x, y = x//base, x%base
        cond = y.reshape(1, -1) == np.arange(0, base, dtype=y.dtype).reshape(base, 1)
        one_hot[idx:idx+base] = np.where(cond, 1.0, 0.0)
        idx -= base
    while idx >= 0:
        one_hot[idx] = 1.0
        idx -= base
    one_hot = one_hot.transpose(1, 0).reshape(*x_shape, len_x*base)
    return one_hot

##############################################
############ OS Function Parts ###############
##############################################
def _get_subdirs(dirpath, leaf_only=True):
    subdirs = list()
    is_leaf = True
    for filename in os.listdir(dirpath):
        if os.path.isdir(os.path.join(dirpath, filename)):
            is_leaf = False
            subdirs.extend(_get_subdirs(os.path.join(dirpath, filename), leaf_only=leaf_only))
    if not leaf_only or is_leaf:
        subdirs.append(dirpath)
    return subdirs

def _read_graphs_from_dir(dirpath):
    import igraph as ig
    graphs = dict()
    for filename in os.listdir(dirpath):
        if not os.path.isdir(os.path.join(dirpath, filename)):
            names = os.path.splitext(os.path.basename(filename))
            if names[1] != ".gml":
                continue
            try:
                graph = ig.read(os.path.join(dirpath, filename))
                graph.vs["label"] = [int(x) for x in graph.vs["label"]]
                # graph.vs["A"] = [int(x) for x in graph.vs["A"]]
                # graph.vs["B"] = [int(x) for x in graph.vs["B"]]
                # graph.vs["C"] = [int(x) for x in graph.vs["C"]]
                # graph.vs["D"] = [int(x) for x in graph.vs["D"]]
                # graph.vs["E"] = [int(x) for x in graph.vs["E"]]
                graph.es["label"] = [int(x) for x in graph.es["label"]]
                graph.es["key"] = [int(x) for x in graph.es["key"]]
                graphs[names[0]] = graph
            except BaseException as e:
                print(e)
                break
    return graphs

def read_graphs_from_dir(dirpath, num_workers=4):
    graphs = dict()
    subdirs = _get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_graphs_from_dir, args=(subdir, ))))
        pool.close()
        
        for subdir, x in tqdm(results):
            x = x.get()
            graphs[os.path.basename(subdir)] = x
    return graphs

def read_patterns_from_dir(dirpath, num_workers=4):
    patterns = dict()
    subdirs = _get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_graphs_from_dir, args=(subdir, ))))
        pool.close()
        
        for subdir, x in tqdm(results):
            x = x.get()
            patterns.update(x)
    return patterns

def _read_metadata_from_dir(dirpath):
    meta = dict()
    for filename in os.listdir(dirpath):
        if not os.path.isdir(os.path.join(dirpath, filename)):
            names = os.path.splitext(os.path.basename(filename))
            if names[1] != ".meta":
                continue
            try:
                with open(os.path.join(dirpath, filename), "r") as f:
                    meta[names[0]] = json.load(f)
            except BaseException as e:
                print(e)
    return meta

def read_metadata_from_dir(dirpath, num_workers=4):
    meta = dict()
    subdirs = _get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_metadata_from_dir, args=(subdir, ))))
        pool.close()
        
        for subdir, x in tqdm(results):
            x = x.get()
            meta[os.path.basename(subdir)] = x
    return meta

def _read_literals_from_dir(dirpath):
    literals = dict()
    for filename in os.listdir(dirpath):
        if not os.path.isdir(os.path.join(dirpath, filename)):
            names = os.path.splitext(os.path.basename(filename))
            if names[1] != ".literals":
                continue
            try:
                with open(os.path.join(dirpath, filename), "r") as f:
                    literals[names[0]] = json.load(f)
            except BaseException as e:
                print(e)
    return literals

def read_literals_from_dir(dirpath, num_workers=4):
    literals = dict()
    subdirs = _get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_literals_from_dir, args=(subdir, ))))
        pool.close()
        
        for subdir, x in tqdm(results):
            x = x.get()
            literals.update(x)
    return literals

def load_data(graph_dir, pattern_dir, metadata_dir, num_workers=4):
    patterns = read_patterns_from_dir(pattern_dir, num_workers=num_workers)
    graphs = read_graphs_from_dir(graph_dir, num_workers=num_workers)
    meta = read_metadata_from_dir(metadata_dir, num_workers=num_workers)
    #add read literals from file
    #literals = read_literals_from_dir(pattern_dir, num_workers=num_workers)

    train_data, dev_data, test_data = list(), list(), list()
    for p, pattern in patterns.items():
        if p in graphs:
            for g, graph in graphs[p].items():
                x = dict()
                x["id"] = ("%s-%s" % (p, g))
                x["pattern"] = pattern
                x["graph"] = graph
                x["subisomorphisms"] = meta[p][g]["subisomorphisms"]
                x["counts"] = meta[p][g]["counts"]
                x["mapping"] = meta[p][g]["mapping"]
                #x["literals"] = literals[p]

                g_idx = int(g.rsplit("_", 1)[-1])
                if g_idx % 10 == 0:
                    dev_data.append(x)
                elif g_idx % 10 == 1:
                    test_data.append(x)
                else:
                    train_data.append(x)
        elif len(graphs) == 1 and "raw" in graphs.keys():
            for g, graph in graphs["raw"].items():
                x = dict()
                x["id"] = ("%s-%s" % (p, g))
                x["pattern"] = pattern
                x["graph"] = graph
                x["subisomorphisms"] = meta[p][g]["subisomorphisms"]
                x["counts"] = meta[p][g]["counts"]
                x["mapping"] = meta[p][g]["mapping"]
                #x["literals"] = literals[p]
                
                g_idx = int(g.rsplit("_", 1)[-1])
                if g_idx % 3 == 0:
                    dev_data.append(x)
                elif g_idx % 3 == 1:
                    test_data.append(x)
                else:
                    train_data.append(x)
    return OrderedDict({"train": train_data, "dev": dev_data, "test": test_data})

def get_best_epochs(log_file):
    regex = re.compile(r"data_type:\s+(\w+)\s+best\s+([\s\w\-]+).*?\(epoch:\s+(\d+)\)")
    best_epochs = dict()
    # get the best epoch
    try:
        lines = subprocess.check_output(["tail", log_file, "-n3"]).decode("utf-8").split("\n")[0:-1]
        print(lines)
    except:
        with open(log_file, "r") as f:
            lines = f.readlines()
    
    for line in lines[-3:]:
        matched_results = regex.findall(line)
        for matched_result in matched_results:
            if "loss" in matched_result[1]:
                best_epochs[matched_result[0]] = int(matched_result[2])
    if len(best_epochs) != 3:
        for line in lines:
            matched_results = regex.findall(line)
            for matched_result in matched_results:
                if "loss" in matched_result[1]:
                    best_epochs[matched_result[0]] = int(matched_result[2])
    return best_epochs