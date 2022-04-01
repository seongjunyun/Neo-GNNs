import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from logger import Logger
from models import NeoGNN, LinkPredictor
from utils import init_seed
import scipy.sparse as ssp
import numpy as np
import os
import pickle
import torch_sparse
import warnings
from utils import AA
from torch_sparse import SparseTensor
from pytorch_indexing import spspmm
from torch_scatter import scatter_add

def train(model, predictor, data, split_edge, optimizer, batch_size,  A, deg, args, epoch):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    count = 0
    for perm, perm_large in zip(DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True), DataLoader(range(pos_train_edge.size(0)), args.gnn_batch_size,
                           shuffle=True)):
        optimizer.zero_grad()
        # compute scores of positive edges
        edge = pos_train_edge[perm].t()
        pos_out, pos_out_struct, pos_out_feat, pos_out_struct_raw = model(edge, data, A, predictor)
        # compute scores of negative edges
        # Just do some trivial random sampling.
        edge_large = pos_train_edge[perm_large].t()
        edge_large = torch.randint(0, data.num_nodes, edge_large.size(), dtype=torch.long,
                            device=edge.device)
        with torch.no_grad():
            h = model.forward_feature(data.x, data.adj_t)
            neg_out_gnn_large = predictor(h[edge_large[0]], h[edge_large[1]])
            neg_large_loss = -torch.log(1 - (torch.sigmoid(neg_out_gnn_large)) + 1e-15)
            edge = edge_large[:,torch.topk(neg_large_loss.squeeze(), batch_size)[1]]
        neg_out, neg_out_struct, neg_out_feat, neg_out_struct_raw = model(edge, data, A, predictor)
        pos_loss = -torch.log(pos_out_struct + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out_struct + 1e-15).mean()
        loss1 = pos_loss + neg_loss
        pos_loss = -torch.log(pos_out_feat + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out_feat + 1e-15).mean()
        loss2 = pos_loss + neg_loss
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss3 = pos_loss + neg_loss
        loss = loss1 + loss2 + loss3 + 1e-3 * (torch.abs(pos_out_struct_raw).mean() + torch.abs(neg_out_struct_raw).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        count += 1
        if count % 50 == 0:
            break

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, A, degree, args):
    model.eval()
    predictor.eval()
    h = model.forward_feature(data.x, data.adj_t)

    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    edge_weight = torch.from_numpy(A.data).to(h.device)
    edge_weight = model.f_edge(edge_weight.unsqueeze(-1))

    row, col = A.nonzero()
    edge_index = torch.stack([torch.from_numpy(row), torch.from_numpy(col)]).type(torch.LongTensor).to(h.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=data.num_nodes)
    deg =  model.f_node(deg).squeeze()

    deg = deg.cpu().numpy()
    A_ = A.multiply(deg).tocsr()
    alpha = torch.softmax(model.alpha, dim=0).cpu()
    pos_valid_preds = []
    pos_valid_preds2 = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        gnn_scores = torch.sigmoid(predictor(h[edge[0]], h[edge[1]]).squeeze()).cpu()
        src, dst = pos_valid_edge[perm].t().cpu()
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
        cur_scores = model.g_phi(cur_scores).squeeze()
        cur_scores = torch.sigmoid(cur_scores).cpu()
        pos_valid_preds2 += [cur_scores]
        cur_scores = alpha[0]*cur_scores + alpha[1] * gnn_scores
        pos_valid_preds += [cur_scores]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    neg_valid_preds = []
    neg_valid_preds2 = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        gnn_scores = torch.sigmoid(predictor(h[edge[0]], h[edge[1]]).squeeze()).cpu()
        src, dst = neg_valid_edge[perm].t().cpu()
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device) 
        cur_scores = model.g_phi(cur_scores).squeeze()
        cur_scores = torch.sigmoid(cur_scores).cpu()
        neg_valid_preds2 += [cur_scores]
        cur_scores = alpha[0]*cur_scores + alpha[1] * gnn_scores
        neg_valid_preds += [cur_scores]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)
    pos_test_preds = []
    pos_test_preds2 = []
    pos_test_preds_gnn = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        gnn_scores = torch.sigmoid(predictor(h[edge[0]], h[edge[1]]).squeeze()).cpu()
        pos_test_preds_gnn += [gnn_scores]
        src, dst = pos_test_edge[perm].t().cpu()
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)     
        cur_scores = model.g_phi(cur_scores).squeeze()
        cur_scores = torch.sigmoid(cur_scores).cpu()
        pos_test_preds2 += [cur_scores]
        cur_scores = alpha[0]*cur_scores + alpha[1] * gnn_scores
        pos_test_preds += [cur_scores]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    neg_test_preds = []
    neg_test_preds2 = []
    neg_test_preds_gnn = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        gnn_scores = torch.sigmoid(predictor(h[edge[0]], h[edge[1]]).squeeze()).cpu()
        neg_test_preds_gnn += [gnn_scores]
        src, dst = neg_test_edge[perm].t().cpu()
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
        cur_scores = model.g_phi(cur_scores).squeeze()
        cur_scores = torch.sigmoid(cur_scores).cpu()
        neg_test_preds2 += [cur_scores]
        cur_scores = alpha[0]*cur_scores + alpha[1] * gnn_scores
        neg_test_preds += [cur_scores]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = 0
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    warnings.simplefilter("ignore", UserWarning)
    parser = argparse.ArgumentParser(description='OGBL-PPA (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1024) 
    parser.add_argument('--gnn_batch_size', type=int, default= 64 * 1024)
    parser.add_argument('--test_batch_size', type=int, default=1024 * 64) 
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)

    parser.add_argument('--f_edge_dim', type=int, default=8) 
    parser.add_argument('--f_node_dim', type=int, default=64) 
    parser.add_argument('--g_phi_dim', type=int, default=64) 

    parser.add_argument('--gnn', type=str, default='NeoGNN')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--alpha', type=float, default=-1)
    parser.add_argument('--beta', type=float, default=0)
    

    args = parser.parse_args()
    print(args)

    args.dataset = 'ppa'

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ppa', root='../dataset',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.x = data.x.to(torch.float)
    if args.use_node_embedding:
        data.x = torch.cat([data.x, torch.load('embedding.pt')], dim=-1)
    data.adj_t = data.adj_t.to_symmetric()
    row, col, _ = data.adj_t.coo()
    edge_index = torch.stack([row,col])
    split_edge = dataset.get_edge_split()
    args.num_train_edges = split_edge['train']['edge'].shape[0]
    data = data.to(device)
    
    model = NeoGNN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout, args=args).to(device)  


    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ppa')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }
    
    # edge_weight = torch.ones(edge_index.size(1), dtype=float)
    edge_weight = torch.ones(edge_index.size(1))
    A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                       shape=(data.num_nodes, data.num_nodes))
    degree = torch.from_numpy(A.sum(axis=0)).squeeze()

    for run in range(args.runs):
        best_valid_performance = 0
        print('#################################          ', run, '          #################################')
        init_seed(run)
        model.reset_parameters()
        predictor.reset_parameters()

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge, optimizer,
                        args.batch_size, A, degree, args, epoch)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, split_edge, evaluator,
                            args.test_batch_size, A, degree, args)
                torch.cuda.empty_cache()
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                    print('---')
                
                valid_performance = results['Hits@100'][1]
                if valid_performance > best_valid_performance:
                        best_valid_performance = valid_performance
        
        for key in loggers.keys():
            print(key)
            final_test, highest_valid = loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        final_mean, final_std = loggers[key].print_statistics()

if __name__ == "__main__":
    main()
