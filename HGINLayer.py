import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl.utils import expand_as_pair
from dgl.nn.pytorch import TypedLinear
from utils import get_act, get_norm


class HGIN(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_layer, dropout=0, 
                 norm=None, res=False, act=None, init_alpha=0, learn_alpha=False, homo=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = tc.nn.Parameter(tc.FloatTensor([init_alpha]), requires_grad=learn_alpha)
        self.mlp = MLP(mlp_layer, input_dim, output_dim, output_dim)
        self.norm = norm
        if norm:
            self.norm_layer = get_norm(norm, output_dim)
            self.res_norm_layer = get_norm(norm, input_dim)
        self.res = res
        self.act = act
        if act:
            self.act_func = get_act(act)
        self.drop = nn.Dropout(dropout)
        if input_dim != output_dim:
            self.res = False
        self.homo = homo
        self.reset_parameters()
    
    def reset_parameters(self):
        for name, layer in self.named_children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # uniform / normal, default: kaiming-uniform
    
    def message(self, edges):
        """user-defined Message function."""
        m = edges.src['h']
        if self.homo:
            dst = edges.data['dst']
            m_dst = tc.cat((m, dst.view(-1, 1)), dim=1)
            m_dst_u, indices, _ = tc.unique(m_dst, dim=0, return_inverse=True, sorted=False, return_counts=True)
            indices = indices.new_empty(m_dst_u.size(0)).scatter_(dim=0, index=indices, src=tc.arange(indices.size(0), dtype=indices.dtype, device=indices.device))
            zero_mask = tc.zeros(m.size(0), device=m.device).view(-1, 1)
            zero_mask[indices] = 1
            m.masked_fill_(zero_mask == 0, 0.0)
        return {'m': m}

    def forward(self, g: dgl.DGLGraph, nfeats, isBlock=False):
        with g.local_scope():
            if isBlock:
                feat_in = nfeats[:g.number_of_dst_nodes()]
                feat_src = nfeats
            else:
                feat_in, feat_src = expand_as_pair(nfeats, g)
            g.srcdata['h'] = feat_src
            g.update_all(self.message, fn.sum('m', 'inneigh'))
            h = (1 + self.alpha) * feat_in + g.dstdata['inneigh']
            del feat_src
            h = self.mlp(h)
            if self.norm:
                h = self.norm_layer(h)
                if self.res and self.training:
                    h = self.res_norm_layer(feat_in) + h
            else:
                if self.res and self.training:
                    h = feat_in + h
            del feat_in
            if self.act:
                h = self.act_func(h)
            h = self.drop(h)
            return h

class HIDGIN(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_layer, dropout=0, 
                 norm=None, res=False, act=None, init_alpha=0, learn_alpha=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = tc.nn.Parameter(tc.FloatTensor([init_alpha]), requires_grad=learn_alpha)
        self.mlp = MLP(mlp_layer, input_dim, output_dim, output_dim)
        self.mlp_id = MLP(mlp_layer, input_dim, output_dim, output_dim)
        self.norm = norm
        if norm:
            self.norm_layer = get_norm(norm, output_dim)
            self.res_norm_layer = get_norm(norm, input_dim)
        self.res = res
        self.act = act
        if act:
            self.act_func = get_act(act)
        self.drop = nn.Dropout(dropout)
        if input_dim != output_dim:
            self.res = False
        self.reset_parameters()
    
    def reset_parameters(self):
        for name, layer in self.named_children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # uniform / normal, default: kaiming-uniform
    
    def message(self, edges):
        """user-defined Message function."""
        m = edges.src['h']
        dst = edges.data['dst']
        m_dst = tc.cat((m, dst.view(-1, 1)), dim=1)
        m_dst_u, indices, _ = tc.unique(m_dst, dim=0, return_inverse=True, sorted=False, return_counts=True)
        indices = indices.new_empty(m_dst_u.size(0)).scatter_(dim=0, index=indices, src=tc.arange(indices.size(0), dtype=indices.dtype, device=indices.device))
        zero_mask = tc.zeros(m.size(0), device=m.device).view(-1, 1)
        zero_mask[indices] = 1
        m.masked_fill_(zero_mask == 0, 0.0)
        return {'m': m}

    def forward(self, g: dgl.DGLGraph, nfeats, isBlock=False):
        with g.local_scope():
            if isBlock:
                feat_in = nfeats[:g.number_of_dst_nodes()]
                feat_src = nfeats
            else:
                feat_in, feat_src = expand_as_pair(nfeats, g)
            g.srcdata['h'] = feat_src
            g.update_all(self.message, fn.sum('m', 'inneigh'))
            h = (1 + self.alpha) * feat_in + g.dstdata['inneigh']
            del feat_src
            id = tc.nonzero(g.ndata['id'] == 1).view(-1)
            h0 = tc.zeros_like(h)
            id_mask = tc.zeros(h.size(0), dtype=tc.bool)
            id_mask[id] = True
            h0[id_mask] = self.mlp_id(h[id_mask])
            h0[~id_mask] = self.mlp(h[~id_mask])
            h = h0
            if self.norm:
                h = self.norm_layer(h)
                if self.res and self.training:
                    h = self.res_norm_layer(feat_in) + h
            else:
                if self.res and self.training:
                    h = feat_in + h
            del feat_in
            if self.act:
                h = self.act_func(h)
            h = self.drop(h)
            return h

        
class RGIN(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_layer, dropout=0, 
                 num_rels=1, regularizer=None, num_bases=None, 
                 norm=None, res=False, act=None, 
                 homo=False, init_alpha=0, learn_alpha=False):
        super().__init__()
        if regularizer is not None and num_bases is None:
            num_bases = num_rels
        self.linear_r = TypedLinear(input_dim, output_dim, num_rels, regularizer, num_bases)
        self.alpha = tc.nn.Parameter(tc.FloatTensor([init_alpha]), requires_grad=learn_alpha)
        self.w1 = nn.Linear(input_dim, input_dim)
        self.mlp = MLP(mlp_layer, input_dim, output_dim, output_dim)
        self.norm = norm
        if norm:
            self.norm_layer = get_norm(norm, output_dim)
        self.res = res
        self.act = act
        if act:
            self.act_func = get_act(act)
        self.drop = nn.Dropout(dropout)
        if input_dim != output_dim:
            self.res = False
        self.homo = homo
        self.reset_parameters()

    def reset_parameters(self):
        for name, layer in self.named_children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # uniform / normal, default: kaiming-uniform

    def message(self, edges):
        """user-defined Message function."""
        m = self.linear_r(edges.src['h'], edges.data['etype'], False)
        if self.homo:
            dst = edges.data['dst']
            msg = tc.cat((m, edges.data['etype'].view(-1, 1), dst.view(-1, 1)), dim=1)
            output, indices, count = tc.unique(msg, dim=0, return_inverse=True, sorted=False, return_counts=True)
            indices = indices.new_empty(output.size(0)).scatter_(0, indices, tc.arange(indices.size(0), dtype=indices.dtype, device=indices.device))
            zero_mask = tc.zeros(m.size(0), device=m.device).view(-1, 1)
            zero_mask[indices] = 1
            zero_mask = (zero_mask == 0)
            m.masked_fill_(zero_mask, 0.0)
        return {'m' : m}

    # def reduce(self, nodes):
    #     """user-defined Reduce function"""
    #     for i in range(nodes.batch_size()):
    #         nodes.mailbox['m'][i] 
    #     return {'h': tc.stack([tc.unique(nodes.mailbox['m'][i], dim=0).sum(0) for i in range(nodes.batch_size())])}

    def forward(self, g, nfeats, efeats):
        with g.local_scope():
            g.edata['etype'] = efeats
            feat_in, feat_mut = expand_as_pair(nfeats, g)
            feat_src, feat_dst = expand_as_pair(feat_mut, g)
            g.srcdata['h'] = feat_src
            g.update_all(self.message, fn.sum('m', 'inneigh'))
            if self.learn_alpha:
                h = (1 + self.alpha) * feat_in + g.dstdata['inneigh']
            else:
                h = self.w1(feat_in) + g.dstdata['inneigh']
            h = self.mlp(h)
            if self.res:
                h = feat_in + h
            if self.norm:
                h = self.norm_layer(h)  
            if self.act:
                h = self.act_func(h)
            h = self.drop(h)
            return h

class RGCN(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer=None,
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.0,
                 layer_norm=False):
        super().__init__()
        if regularizer is not None and num_bases is None:
            num_bases = num_rels
        self.linear_r = TypedLinear(in_feat, out_feat, num_rels, regularizer, num_bases)
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(tc.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # TODO(minjie): consider remove those options in the future to make
        #   the module only about graph convolution.
        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(tc.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def message(self, edges):
        """user-defined Message function."""
        m = self.linear_r(edges.src['h'], edges.data['etype'], self.presorted)
        # print(edges._graph)
        # print(edges._graph.edata['_ID'])
        # print(edges._eid)
        dst = edges.data['dst']
        # print(dst)
        # distinct
        msg = tc.cat((m, edges.data['etype'].view(-1, 1), dst.view(-1, 1)), dim=1)
        output, indices, count = tc.unique(msg, dim=0, return_inverse=True, sorted=False, return_counts=True)
        indices = indices.new_empty(output.size(0)).scatter_(0, indices, tc.arange(indices.size(0), dtype=indices.dtype, device=indices.device))
        zero_mask = tc.zeros(m.size(0), device=m.device).view(-1, 1)
        zero_mask[indices] = 1
        zero_mask = (zero_mask == 0)
        m.masked_fill_(zero_mask, 0.0)
        if 'norm' in edges.data:
            m = m * edges.data['norm']
        return {'m' : m}

    # def reduce(self, nodes):
    #     """user-defined Reduce function"""
    #     for i in range(nodes.batch_size()):
    #         nodes.mailbox['m'][i] 
    #     return {'h': th.stack([th.unique(nodes.mailbox['m'][i], dim=0).sum(0) for i in range(nodes.batch_size())])}

    def forward(self, g, feat, etypes, norm=None, *, presorted=False):
        """Forward computation.
        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        etypes : torch.Tensor or list[int]
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        norm : torch.Tensor, optional
            An 1D tensor of edge norm value.  Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether the edges of the input graph have been sorted by their types.
            Forward on pre-sorted graph may be faster. Graphs created
            by :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for sorting edges manually.
        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{out})`.
        """
        self.presorted = presorted
        with g.local_scope():
            g.srcdata['h'] = feat
            if norm is not None:
                g.edata['norm'] = norm
            g.edata['etype'] = etypes
            # message passing
            g.update_all(self.message, fn.sum('m', 'h'))
            # apply bias and activation
            h = g.dstdata['h']
            if self.layer_norm:
                h = self.layer_norm_weight(h)
            if self.bias:
                h = h + self.h_bias
            if self.self_loop:
                h = h + feat[:g.num_dst_nodes()] @ self.loop_weight
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h
        
class AGGR(nn.Module):
    def __init__(self, input_dim, feat_dim, norm, act, res):
        super().__init__()
        self.linear1 = nn.Linear(feat_dim * 2, feat_dim)
        self.linear2 = nn.Linear(feat_dim, feat_dim)
        self.norm = norm
        self.act = act
        self.res = res
        if input_dim != feat_dim:
            self.res = False
        if norm:
            self.norm_layer = get_norm(norm, feat_dim)
        if act:
            self.act_func = get_act(act)
        # self.reset_parameters()

    def forward(self, a, b, feat_in):
        h = self.linear1(tc.cat((a, b), dim=1))
        h = self.act_func(h)
        h = self.linear2(h)
        if self.res:
            h = feat_in + h
        if self.norm:
            h = self.norm_layer(h)
        if self.act:
            h = self.act_func(h)
        return h
    
    def reset_parameters(self):
        for name, layer in self.named_children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model 1 layer
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = tc.nn.ModuleList()
            self.batch_norms = tc.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def reset_parameters(self):
        if self.linear_or_not:
            self.linear.reset_parameters()
        else:
            for name, layer in self.named_children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            for name, linear in self.linears.named_modules():
                if hasattr(linear, 'reset_parameters'):
                    linear.reset_parameters()

    def forward(self, x):
        if self.linear_or_not:
            # If linear model 1 layer
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                # h = F.relu(self.batch_norms[i](self.linears[i](h)))
                h = F.relu(self.linears[i](h))
            return self.linears[-1](h)


class ApplyNodeFunc(nn.Module):
    """
        This class is used in class GINNet
        Update the node feature hv with MLP
    """

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, h):
        h = self.mlp(h)
        return 
