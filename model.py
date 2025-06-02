from cgi import print_arguments
import grp
from random import random
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import math
from dgl.nn.pytorch import RelGraphConv, GINConv
from disrelgraphconv import DisRelGraphConv
from HGINLayer import ApplyNodeFunc, MLP, HGIN, HIDGIN, AGGR
from utils import BinaryEncoding, int2onehot, get_enc_len, identity_feature, anchor_ego_net
from embedding import OrthogonalEmbedding, NormalEmbedding, EquivariantEmbedding
import matplotlib.pyplot as plt

class RGINLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, regularizer="basis", num_bases=None, dropout=0.0, dis=False):
        super(RGINLayer, self).__init__()
        if dis:
            self.rgc_layer = DisRelGraphConv(
                in_feat=in_feat, out_feat=out_feat, num_rels=num_rels,
                regularizer=regularizer, num_bases=num_bases,
                activation=None, self_loop=True, dropout=0.0)
        else:
            self.rgc_layer = RelGraphConv(
                in_feat=in_feat, out_feat=out_feat, num_rels=num_rels,
                regularizer=regularizer, num_bases=num_bases,
                activation=None, self_loop=True, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(out_feat, out_feat),
            nn.ReLU(), 
            nn.Linear(out_feat, out_feat),
            nn.ReLU())
        self.drop = nn.Dropout(dropout)
        
        if hasattr(self.rgc_layer, "weight") and self.rgc_layer.weight is not None:
            nn.init.normal_(self.rgc_layer.weight, 0.0, 1/(out_feat)**0.5)
        if hasattr(self.rgc_layer, "w_comp") and self.rgc_layer.w_comp is not None:
            nn.init.normal_(self.rgc_layer.w_comp, 0.0, 1/(out_feat)**0.5)
        if hasattr(self.rgc_layer, "loop_weight") and self.rgc_layer.loop_weight is not None:
            nn.init.normal_(self.rgc_layer.loop_weight, 0.0, 1/(out_feat)**0.5)
        if hasattr(self.rgc_layer, "h_bias") and self.rgc_layer.h_bias is not None:
            nn.init.zeros_(self.rgc_layer.h_bias)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 1/(out_feat)**0.5)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, g, x, etypes):
        g = self.rgc_layer(g, x, etypes, norm=None)
        g = self.mlp(g)
        g = self.drop(g)
        return g

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config
        self.init_emb = config["init_emb"]
        self.base = config["base"]

        self.max_ngv = config["max_ngv"]
        self.max_ngvl = config["max_ngvl"]
        self.max_nge = config["max_nge"]
        self.max_ngel = config["max_ngel"]
        
        self.dropout = config["dropout"]

        self.emb_out_dim = config["net_hidden_dim"]
        self.layer_type = config["layer_type"]
        self.num_layers = config["num_layers"]
        self.net_hidden_dim = config["net_hidden_dim"]
        self.conv = config["conv"]
        self.predict_hidden_dim = config["predict_hidden_dim"]
        self.bidirectional = config["bidirectional"]
        self.add_feat_enc = config["add_feat_enc"]
        self.add_id_enc = config["add_id_enc"]
        self.add_pos_enc = config["add_pos_enc"]
        self.order_emb = config["order_emb"]
        self.margin = config["margin"]
        self.skip_layer = config["skip_layer"]
        self.abs = config["abs"]
        self.gpu_id = config["gpu_id"]


        self.g_vl_enc = BinaryEncoding(self.max_ngvl)
        self.p_vl_enc = self.g_vl_enc

        self.g_vl_emb = self.create_emb(self.g_vl_enc.out_len, self.emb_out_dim, init_emb=self.init_emb)
        self.p_vl_emb = self.g_vl_emb
        config["llm_emb_file"], config["llm_emb_dim"] = get_llm_info(config["llm"])
        p_emb_out_dim, g_emb_out_dim = self.get_emb_out_dim()
        self.llm_emb = nn.Linear(config["llm_emb_dim"], self.emb_out_dim)
        

        if self.add_id_enc:
            id_enc_dim = self.config["id_enc_dim"]
            p_emb_out_dim += id_enc_dim
            g_emb_out_dim += id_enc_dim
        
        if self.layer_type == "hgin":
            self.g_net, g_dim = self.create_hgin_net(
                name="graph", net_in_dim=g_emb_out_dim, net_hidden_dim=self.net_hidden_dim,
                num_layers=config["num_layers"], num_mlp_layers=config["num_mlp_layers"], 
                layer_aggr_type=config["layer_aggr_type"], dropout=config["dropout"], 
                init_eps=config["init_eps"], learn_eps=config["learn_eps"], homo=config["homo"])
        elif self.layer_type == "dhgin":
            self.g_net, g_dim = self.create_dhgin_net(
                name="graph", net_in_dim=g_emb_out_dim, net_hidden_dim=self.net_hidden_dim,
                num_layers=config["num_layers"], num_mlp_layers=config["num_mlp_layers"], 
                layer_aggr_type=config["layer_aggr_type"], dropout=config["dropout"],  batch_norm=config["batch_norm"], 
                residual=config["residual"], init_eps=config["init_eps"], learn_eps=config["learn_eps"])
        elif self.layer_type == "rgin":
            self.g_net, g_dim = self.create_rgin_net(
                name="graph", input_dim=g_emb_out_dim, hidden_dim=self.net_hidden_dim,
                num_layers=config["num_layers"], num_rels=self.max_ngel, 
                num_bases=config["rgin_num_bases"], regularizer=config["rgin_regularizer"], 
                dropout=self.dropout)
        else: 
            raise NotImplementedError
        self.p_net, p_dim = self.g_net, g_dim

        if self.bidirectional:
            self.aggrs = nn.ModuleList()
            self.aggrs.add_module(f"AGGR_1", AGGR(input_dim=g_emb_out_dim, feat_dim=self.net_hidden_dim, norm=None, act='relu', res=True))
            for i in range(config['num_layers'] - 1):
                self.aggrs.add_module(f"AGGR_{i+2}", AGGR(input_dim=self.net_hidden_dim, feat_dim=self.net_hidden_dim, norm=None, act='relu', res=True))
        
        if config["aggr_norm"]:
            self.aggr_norm = nn.LayerNorm(p_dim)

        if self.skip_layer:
            self.post_mp = nn.Sequential(
                nn.Linear(p_dim * (self.num_layers + 1), p_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(p_dim, p_dim),
                nn.ReLU(),
                nn.Linear(p_dim, 256), nn.ReLU(),
                nn.Linear(256, p_dim))

        if self.add_feat_enc:
            p_enc_dim, g_enc_dim = self.get_feat_enc_dim()
            p_dim += p_enc_dim
            g_dim += g_enc_dim

        self.clf_model = nn.Sequential(
            nn.Linear(1, 2), 
            nn.LogSoftmax(dim=-1))

    def create_enc(self, max_n, base):
        enc_len = get_enc_len(max_n-1, base)
        enc_dim = enc_len * base
        enc = nn.Embedding(max_n, enc_dim)
        enc.weight.data.copy_(torch.from_numpy(int2onehot(np.arange(0, max_n), enc_len, base)))
        enc.weight.requires_grad=False
        return enc

    def create_emb(self, emb_in_dim, emb_out_dim, init_emb="Orthogonal"):
        if init_emb == "None" or init_emb == 'llm':
            emb = None
        elif init_emb == "Orthogonal":
            emb = OrthogonalEmbedding(emb_in_dim, emb_out_dim)
        elif init_emb == "Normal":
            emb = NormalEmbedding(emb_in_dim, emb_out_dim)
        elif init_emb == "Equivariant":
            emb = EquivariantEmbedding(emb_in_dim, emb_out_dim)
        elif init_emb == "Position":
            print("init emb=position")
            emb = PGNNEmbedding(emb_in_dim, emb_out_dim)
            print("embbeding created!!!")
        else:
            raise NotImplementedError
        return emb

    def create_hgin_net(self, name, net_in_dim, **kw):
        net_hidden_dim = kw.get("net_hidden_dim", 64)
        num_layers = kw.get("num_layers", 1)
        num_mlp_layers = kw.get("num_mlp_layers", 1)
        aggr_type = kw.get("layer_aggr_type", "sum")
        dropout = kw.get("dropout", 0.0)
        batch_norm = kw.get("batch_norm", None)
        residual = kw.get("residual", False)
        act = kw.get("act", None)
        init_eps = kw.get("init_eps", 0.001)
        learn_eps = kw.get("learn_eps", False)
        homo = kw.get("homo", True)
        hgins = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = net_in_dim
            if i > 0:
                if self.skip_layer:
                    layer_input_dim = net_hidden_dim * (i + 1)
                else:
                    layer_input_dim = net_hidden_dim
            if self.config['ego']:
                hgins.add_module(f"{name}_hidgin{i}", HIDGIN(
                    input_dim=net_hidden_dim if i > 0 else net_in_dim, output_dim=net_hidden_dim, 
                    mlp_layer=num_mlp_layers, dropout=dropout, norm=batch_norm, res=residual, 
                    act=act, init_alpha=init_eps, learn_alpha=learn_eps))
            else:
                hgins.add_module(f"{name}_hgin{i}", HGIN(
                    input_dim=net_hidden_dim if i > 0 else net_in_dim, output_dim=net_hidden_dim, 
                    mlp_layer=num_mlp_layers, dropout=dropout, norm=batch_norm, res=residual, 
                    act=act, init_alpha=init_eps, learn_alpha=learn_eps, homo=homo))
            
        return hgins, net_hidden_dim
    
    def create_dhgin_net(self, name, net_in_dim, **kw):
        net_hidden_dim = kw.get("net_hidden_dim", 64)
        num_layers = kw.get("num_layers", 1)
        num_mlp_layers = kw.get("num_mlp_layers", 1)
        aggr_type = kw.get("layer_aggr_type", "sum")
        dropout = kw.get("dropout", 0.0)
        batch_norm = kw.get("batch_norm", False)
        residual = kw.get("residual", False)
        init_eps = kw.get("init_eps", 0.001)
        learn_eps = kw.get("learn_eps", True)
        hgins = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = net_in_dim
            if i > 0:
                if self.skip_layer:
                    layer_input_dim = net_hidden_dim * (i + 1)
                else:
                    layer_input_dim = net_hidden_dim
            mlp = MLP(num_mlp_layers, layer_input_dim, net_hidden_dim, net_hidden_dim)
            hgins.add_module(f"{name}_hgin{i}", DHGINLayer(
                apply_func=ApplyNodeFunc(mlp), aggr_type=aggr_type, dropout=dropout, batch_norm=batch_norm,
                residual=residual, init_eps=init_eps, learn_eps=learn_eps))
            
        return hgins, net_hidden_dim
    
    def create_rgin_net(self, name, input_dim, **kw):
        num_layers = kw.get("num_layers", 1)
        hidden_dim = kw.get("hidden_dim", 64)
        num_rels = kw.get("num_rels", 1)
        num_bases = kw.get("num_bases", 8)
        regularizer = kw.get("regularizer", "basis")
        dropout = kw.get("dropout", 0.0)
        self.dis = False
        if self.conv == 'DisRGIN':
            self.dis = True
        rgins = nn.ModuleList()
        for i in range(num_layers):
            if self.skip_layer:
                hidden_input_dim = hidden_dim * (i + 1)
            else:
                hidden_input_dim = hidden_dim
            rgins.add_module(f"{name}_rgi{i}", RGINLayer(
                in_feat=hidden_input_dim if i > 0 else input_dim, out_feat=hidden_dim, num_rels=num_rels,
                regularizer=regularizer, num_bases=num_bases, dropout=dropout, dis=self.dis))

        return rgins, hidden_dim

    def get_feat_enc_dim(self):
        g_dim = self.g_vl_enc.out_len
        return g_dim, g_dim

    def get_feat_enc(self, pattern, graph):
        pattern_vl = self.p_vl_enc(pattern.ndata["label"])
        graph_vl = self.g_vl_enc(graph.ndata["label"])
        return pattern_vl, graph_vl
    
    def get_id_enc(self, graph, node_list=None):
        return identity_feature(graph, self.config["id_enc_dim"], node_list)
    
    def get_emb_out_dim(self):
        if self.init_emb == "None":
            return self.get_feat_enc_dim()
        elif self.init_emb == 'llm':
            if self.config["llmdim"]:
                self.net_hidden_dim = 256
                self.emb_out_dim = 256
                return self.config["llm_emb_dim"], self.config["llm_emb_dim"]
            else:
                return self.emb_out_dim, self.emb_out_dim
        else:
            return self.emb_out_dim, self.emb_out_dim

    def get_emb(self, pattern, graph):
        pattern_vl = self.p_vl_enc(pattern.ndata["label"])
        graph_vl = self.g_vl_enc(graph.ndata["label"])
        if self.init_emb == "None":
            p_emb = pattern_vl
            g_emb = graph_vl
        elif self.init_emb == 'llm':
            if self.config["llmdim"]:
                p_emb = label2emb(self.config["llm_emb_file"], pattern.ndata["label"])
                g_emb = label2emb(self.config["llm_emb_file"], graph.ndata["label"])
            else:
                p_emb = self.llm_emb(label2emb(self.config["llm_emb_file"], pattern.ndata["label"]))
                g_emb = self.llm_emb(label2emb(self.config["llm_emb_file"], graph.ndata["label"]))
        else:
            p_emb = self.p_vl_emb(pattern_vl)
            g_emb = self.g_vl_emb(graph_vl)

        return p_emb, g_emb
        
    def forward(self, pattern, graph):
        pattern_emb, graph_emb = self.get_emb(pattern, graph)
        if self.add_id_enc:
            pattern_id_enc = self.get_id_enc(pattern)
            graph_id_enc = self.get_id_enc(graph)
            pattern_emb = torch.cat([pattern_emb, pattern_id_enc], dim=1)
            graph_emb = torch.cat([graph_emb, graph_id_enc], dim=1)

        if not self.bidirectional:
            pattern.edata['dst'] = pattern.edges()[1]
            graph.edata['dst'] = graph.edges()[1]
            pattern_output = pattern_emb
            all_emb = pattern_emb
            for p_rgin in self.p_net:
                if self.layer_type == "rgin":
                    o = p_rgin(pattern, pattern_output, pattern.edata["label"])
                elif self.layer_type == "hgin":
                    o = p_rgin(pattern, pattern_output)
                else:
                    raise NotImplementedError
                if self.skip_layer:
                    pattern_output = torch.cat([pattern_output, o], dim=1)
                else:
                    pattern_output = o + pattern_output

            graph_output = graph_emb
            all_emb = graph_emb
            for g_rgin in self.g_net:
                if self.layer_type == "rgin":
                    o = g_rgin(graph, graph_output, graph.edata["label"])
                elif self.layer_type == "hgin":
                    o = g_rgin(graph, graph_output)
                if self.skip_layer:
                    graph_output = torch.cat([graph_output, o], dim=1)
                else:
                    graph_output = o + graph_output
        else:
            rpattern = dgl.reverse(pattern)
            rgraph = dgl.reverse(graph)
            pattern.edata['dst'] = pattern.edges()[1]
            graph.edata['dst'] = graph.edges()[1]
            rpattern.edata['dst'] = rpattern.edges()[1]
            rgraph.edata['dst'] = rgraph.edges()[1]
            pattern_output = pattern_emb
            graph_output = graph_emb
            for rgin, aggr in zip(self.p_net, self.aggrs):
                torch.cuda.empty_cache()
                if self.layer_type == "rgin":
                    o = rgin(pattern, pattern_output, pattern.edata["label"])
                    ro = rgin(rpattern, pattern_output, pattern.edata["label"])
                elif self.layer_type == "hgin":
                    o = rgin(pattern, pattern_output)
                    ro = rgin(rpattern, pattern_output)
                else:
                    raise NotImplementedError
                if self.skip_layer:
                    pattern_output = torch.cat([pattern_output, F.relu(self.aggr(torch.cat([o, ro], dim=1)))], dim=1)
                else: 
                    pattern_output = aggr(o, ro, pattern_output)
                if self.layer_type == 'rgin':
                    o = rgin(graph, graph_output, graph.edata["label"])
                    ro = rgin(rgraph, graph_output, graph.edata["label"])
                elif self.layer_type == "hgin":
                    o = rgin(graph, graph_output)
                    ro = rgin(rgraph, graph_output)
                else:
                    raise NotImplementedError
                if self.skip_layer:
                    graph_output = torch.cat([graph_output, F.relu(self.aggr(torch.cat([o, ro], dim=1)))], dim=1) 
                else:
                    graph_output = aggr(o, ro, graph_output)

        if self.skip_layer:
            pattern_output = self.post_mp(pattern_output)
            graph_output = self.post_mp(graph_output)
        
        if self.add_feat_enc:
            pattern_enc, graph_enc = self.get_feat_enc(pattern, graph)
            pattern_output = torch.cat([pattern_enc, pattern_output], dim=1)
            graph_output = torch.cat([graph_enc, graph_output], dim=1)


        p_u_emb = pattern_output[pattern.ndata["anchored"] == 1]
        g_u_emb = graph_output[graph.ndata["anchored"] == 1]

        if self.abs:
            p_u_emb = torch.abs(p_u_emb)
            g_u_emb = torch.abs(g_u_emb)
        if self.config["nonneg"]:
            p_u_emb = torch.clamp(p_u_emb, min=0)
            g_u_emb = torch.clamp(g_u_emb, min=0)
        return p_u_emb, g_u_emb
    
    def margin_loss(self, emb, label, weight=1):
        margin = self.margin
        p_u_emb, g_u_emb = emb
        e = torch.sum(torch.max(torch.zeros_like(p_u_emb, device=p_u_emb.device), 
                                p_u_emb - g_u_emb)**2, dim=1)
        e[label == 0] = torch.max(torch.tensor(0.0, device=p_u_emb.device), margin - e)[label == 0]
        if weight != 1:
            e[label == 0] = e[label == 0] * weight
        loss = torch.sum(e)
        if self.config["rglr_mlpid"]:
            mlp_l2loss = 0
            for gin in self.p_net:
                id_norm = sum((p.pow(2).sum() + 1e-8).sqrt() for p in gin.mlp_id.parameters())
                mlp_norm = sum((p.pow(2).sum() + 1e-8).sqrt() for p in gin.mlp.parameters())
                mlp_l2loss += 1 / (id_norm - mlp_norm)
            loss += mlp_l2loss
        return loss
    
    def predict(self, emb, pred=None):
        p_u_emb, g_u_emb = emb
        e = torch.sum(torch.max(torch.zeros_like(p_u_emb,
        device=p_u_emb.device), p_u_emb - g_u_emb)**2, dim=-1).view(-1, 1)
        return e

    def emb_graph(self, graph, batch_size=128, device=torch.device('cuda')):
        all_encs = []
        all_embs = []
        store_device=torch.device('cpu')
        num_nodes = graph.number_of_nodes()
        for i in range(0, num_nodes, batch_size):
            start = i
            end = min(i + batch_size, num_nodes)
            batch_labels = graph.ndata['label'][start:end].view(-1, 1).to(device)
            batch_g_vl_enc = self.g_vl_enc(batch_labels)
            all_encs.append(batch_g_vl_enc.to(store_device))
            if self.init_emb == "None":
                batch_g_emb = batch_g_vl_enc
            else:
                batch_g_emb = self.g_vl_emb(batch_g_vl_enc)
            if self.add_id_enc:
                g_id_enc = self.get_id_enc(graph, list(range(start, end, 1))).to(device)
                batch_g_emb = torch.cat([batch_g_emb, g_id_enc], dim=1)
                del g_id_enc
            all_embs.append(batch_g_emb.to(store_device))
            del batch_labels, batch_g_vl_enc, batch_g_emb
            torch.cuda.empty_cache()
        g_enc = torch.cat(all_encs, dim=0).to(store_device)
        g_emb = torch.cat(all_embs, dim=0).to(store_device)

        rgraph = dgl.reverse(graph, copy_edata=True)
        graph.edata['dst'] = graph.edges()[1]
        rgraph.edata['dst'] = rgraph.edges()[1]
        graph_output = g_emb
        torch.cuda.empty_cache()
        layer_num = 1
        for rgin, aggr in zip(self.p_net, self.aggrs):
            layer_num += 1
            y = torch.zeros(graph_output.size(0), rgin.output_dim, device=store_device)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                graph, torch.arange(graph.number_of_nodes(), device=store_device), sampler,
                batch_size=batch_size, shuffle=False, drop_last=False)
            torch.cuda.empty_cache()
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].to(device)
                h = graph_output[input_nodes].to(device)
                if self.layer_type == 'rgin':
                    h = rgin(block, h, block.edata["label"])
                else:
                    h = rgin(block, h, isBlock=True)
                y[output_nodes] = h.cpu()
                del block, h
                torch.cuda.empty_cache()

            ry = torch.zeros(graph_output.size(0), rgin.output_dim, device=store_device)
            rdataloader = dgl.dataloading.DataLoader(
                rgraph, torch.arange(graph.number_of_nodes(), device=store_device), sampler,
                batch_size=batch_size, shuffle=False, drop_last=False)
            torch.cuda.empty_cache()
            for input_nodes, output_nodes, blocks in rdataloader:
                block = blocks[0].to(device)
                h = graph_output[input_nodes].to(device)
                if self.layer_type == 'rgin':
                    h = rgin(block, h, block.edata["label"])
                else:
                    h = rgin(block, h, isBlock=True)
                ry[output_nodes] = h.cpu()
                del block, h
                torch.cuda.empty_cache()
            all_outputs = []
            for i in range(0, num_nodes, batch_size):
                batch_nodes = slice(i, min(i + batch_size, num_nodes))
                batch_y = y[batch_nodes].to(device)
                batch_ry = ry[batch_nodes].to(device)
                batch_output = graph_output[batch_nodes].to(device)
                batch_output = aggr(batch_y, batch_ry, batch_output)
                all_outputs.append(batch_output.to(store_device))
                del batch_nodes, batch_y, batch_ry, batch_output
                torch.cuda.empty_cache()
            graph_output = torch.cat(all_outputs, dim=0).to(store_device)
            del all_outputs
            torch.cuda.empty_cache()
        
        if self.skip_layer:
            graph_output = self.post_mp(graph_output)
        if self.add_feat_enc:
            graph_output = torch.cat([g_enc, graph_output], dim=1)
        if self.abs:
            graph_output = torch.abs(graph_output)
        return graph_output

    def emb_pattern(self, pattern):
        p_vl_enc = self.p_vl_enc(pattern.ndata["label"].view(-1, 1))
        if self.init_emb == "None":
            p_emb = p_vl_enc
        else:
            p_emb = self.p_vl_emb(p_vl_enc)
        if self.add_id_enc:
            p_id_enc = self.get_id_enc(pattern)
            p_emb = torch.cat([p_emb, p_id_enc], dim=1)

        rpattern = dgl.reverse(pattern)
        pattern.edata['dst'] = pattern.edges()[1]
        rpattern.edata['dst'] = rpattern.edges()[1]
        pattern_output = p_emb
        for rgin, aggr in zip(self.p_net, self.aggrs):
            if self.layer_type == 'rgin':
                o = rgin(pattern, pattern_output, pattern.edata["label"])
                ro = rgin(rpattern, pattern_output, pattern.edata["label"])
            else:
                o = rgin(pattern, pattern_output)
                ro = rgin(rpattern, pattern_output)
            if self.skip_layer:
                pattern_output = torch.cat([pattern_output, F.relu(self.aggr(torch.cat([o, ro], dim=1)))], dim=1)
            else:
                pattern_output = aggr(o, ro, pattern_output)
        
        if self.skip_layer:
            pattern_output = self.post_mp(pattern_output)
        if self.add_feat_enc:
            pattern_output = torch.cat([self.p_vl_enc(pattern.ndata["label"]), pattern_output], dim=1)
        if self.abs:
            pattern_output = torch.abs(pattern_output)
        return pattern_output