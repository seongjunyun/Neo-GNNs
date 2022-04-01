from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch import nn
from utils import glorot, zeros
import pdb
from pytorch_indexing import spspmm



class NeoGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super(NeoGNN, self).__init__()

        self.args = args
        self.convs = torch.nn.ModuleList()
        if args.dataset == 'citation2':
            cached = False
        else:
            cached = True
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=cached))

        self.dropout = dropout
        self.alpha = torch.nn.Parameter(torch.FloatTensor([0, 0]))

        if args.dataset not in ['ppa', 'citation2']:
            self.f_edge = torch.nn.Sequential(torch.nn.Linear(1, args.f_edge_dim).double(),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(args.f_edge_dim, 1).double())

            self.f_node = torch.nn.Sequential(torch.nn.Linear(1, args.f_node_dim).double(),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(args.f_node_dim, 1).double())

            self.g_phi = torch.nn.Sequential(torch.nn.Linear(1, args.g_phi_dim).double(),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(args.g_phi_dim, 1).double())
        else:
            self.f_edge = torch.nn.Sequential(torch.nn.Linear(1, args.f_edge_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(args.f_edge_dim, 1))

            self.f_node = torch.nn.Sequential(torch.nn.Linear(1, args.f_node_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(args.f_node_dim, 1))

            self.g_phi = torch.nn.Sequential(torch.nn.Linear(1, args.g_phi_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(args.g_phi_dim, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.constant_(self.alpha, 0)
        self.f_edge.apply(self.weight_reset)
        self.f_node.apply(self.weight_reset)
        self.g_phi.apply(self.weight_reset)

    def weight_reset(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    
    def forward(self, edge, data, A, predictor=None, emb=None, only_feature=False, only_structure=False, node_struct_feat=None):
        batch_size = edge.shape[-1]
        # 1. compute similarity scores of node pairs via conventionl GNNs (feature + adjacency matrix)
        adj_t = data.adj_t
        out_feat = None
        if not only_structure:
            if emb is None:
                x = data.x
            else:
                x = emb
            for conv in self.convs[:-1]:
                x = conv(x, adj_t)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, adj_t)
            if predictor is not None:
                out_feat = predictor(x[edge[0]], x[edge[1]])
            else:
                out_feat = torch.sum(x[edge[0]] * x[edge[1]], dim=0)
        
        if only_feature:
            return None, None, out_feat
        # 2. compute similarity scores of node pairs via Neo-GNNs
        # 2-1. Structural feature generation
        if node_struct_feat is None:
            row_A, col_A = A.nonzero()
            tmp_A = torch.stack([torch.from_numpy(row_A), torch.from_numpy(col_A)]).type(torch.LongTensor).to(edge.device)
            row_A, col_A = tmp_A[0], tmp_A[1]
            edge_weight_A = torch.from_numpy(A.data).to(edge.device)
            edge_weight_A = self.f_edge(edge_weight_A.unsqueeze(-1))
            node_struct_feat = scatter_add(edge_weight_A, col_A, dim=0, dim_size=data.num_nodes)

        indexes_src = edge[0].cpu().numpy()
        row_src, col_src = A[indexes_src].nonzero()
        edge_index_src = torch.stack([torch.from_numpy(row_src), torch.from_numpy(col_src)]).type(torch.LongTensor).to(edge.device)
        edge_weight_src = torch.from_numpy(A[indexes_src].data).to(edge.device)
        edge_weight_src = edge_weight_src * self.f_node(node_struct_feat[col_src]).squeeze()

        indexes_dst = edge[1].cpu().numpy()
        row_dst, col_dst = A[indexes_dst].nonzero()
        edge_index_dst = torch.stack([torch.from_numpy(row_dst), torch.from_numpy(col_dst)]).type(torch.LongTensor).to(edge.device)
        edge_weight_dst = torch.from_numpy(A[indexes_dst].data).to(edge.device)
        edge_weight_dst = edge_weight_dst * self.f_node(node_struct_feat[col_dst]).squeeze()
        
        
        if self.args.dataset in ['ppa', 'citation2']:
            edge_index_dst = torch.stack([edge_index_dst[1], edge_index_dst[0]])
            edge_indexes, scores = spspmm(edge_index_src, edge_weight_src, edge_index_dst, edge_weight_dst, batch_size, data.num_nodes, batch_size, data_split=256)
            out_struct = torch.zeros(batch_size).to(edge.device)
            out_struct[edge_indexes[0][edge_indexes[0]==edge_indexes[1]]] = scores[edge_indexes[0]==edge_indexes[1]]
        else:
            mat_src = SparseTensor.from_edge_index(edge_index_src, edge_weight_src, [batch_size, data.num_nodes])
            mat_dst = SparseTensor.from_edge_index(edge_index_dst, edge_weight_dst, [batch_size, data.num_nodes])
            out_struct = (mat_src @ mat_dst.to_dense().t()).diag()
        
        out_struct = self.g_phi(out_struct.unsqueeze(-1))
        out_struct_raw = out_struct
        out_struct = torch.sigmoid(out_struct)

        if not only_structure:
            alpha = torch.softmax(self.alpha, dim=0)
            out = alpha[0] * out_struct + alpha[1] * out_feat + 1e-15
        else:
            out = None

        del edge_weight_src, edge_weight_dst, node_struct_feat
        torch.cuda.empty_cache()

        return out, out_struct, out_feat, out_struct_raw

    def forward_feature(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        if args.dataset == 'citation2':
            cached = False
        else:
            cached = True
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=cached))

        self.dropout = dropout

    def reset_parameters(self, heuristic=False):
        
        if not heuristic:
            for conv in self.convs:
                conv.reset_parameters()

    def weight_reset(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    
    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

        self.alpha = Parameter(torch.Tensor(1))
        self.theta = Parameter(torch.Tensor(1))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        
        torch.nn.init.constant_(self.alpha, 0.5)
        torch.nn.init.constant_(self.theta, 2)

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    

class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        value = adj_t.storage.value()
        if value is not None and adj_t.storage.value().requires_grad == False:
            adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)








@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]




class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.
    Its node-wise formulation is given by:
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j
    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



