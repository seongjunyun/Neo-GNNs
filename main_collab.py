import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from logger import Logger
from models import NeoGNN, LinkPredictor
from utils import init_seed
import numpy as np
import scipy.sparse as ssp
import os
import pickle
import torch_sparse
import warnings
from utils import AA
from torch_sparse import SparseTensor
from torch_scatter import scatter_add
    
def train(model, predictor, data, split_edge, optimizer, batch_size, A, deg, args):
    model.train()
    predictor.train()
    
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    total_loss = total_examples = 0
    count = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        # compute scores of positive edges
        edge = pos_train_edge[perm].t()
        pos_out, pos_out_struct, pos_out_feat, _ = model(edge, data, A, predictor)
        
        # compute scores of negative edges
        # Just do some trivial random sampling.
        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                            device=edge.device)
        neg_out, neg_out_struct, neg_out_feat, _ = model(edge, data, A, predictor)

        pos_loss = -torch.log(pos_out_struct + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out_struct + 1e-15).mean()
        loss1 = pos_loss + neg_loss
        pos_loss = -torch.log(pos_out_feat + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out_feat + 1e-15).mean()
        loss2 = pos_loss + neg_loss
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss3 = pos_loss + neg_loss
        loss = loss1 + loss2 + loss3
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

    pos_train_edge = split_edge['train']['edge'].to(h.device)
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
    print(alpha)
    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        gnn_scores = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        src, dst = pos_train_edge[perm].t().cpu()
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
        cur_scores = torch.sigmoid(model.g_phi(cur_scores).squeeze().cpu())  
        cur_scores = alpha[0]*cur_scores + alpha[1] * gnn_scores
        pos_train_preds += [cur_scores]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)
    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        gnn_scores = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        src, dst = pos_valid_edge[perm].t().cpu()
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
        cur_scores = torch.sigmoid(model.g_phi(cur_scores).squeeze().cpu())     
        cur_scores = alpha[0]*cur_scores + alpha[1] * gnn_scores
        pos_valid_preds += [cur_scores]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        gnn_scores = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        src, dst = neg_valid_edge[perm].t().cpu()
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
        cur_scores = torch.sigmoid(model.g_phi(cur_scores).squeeze().cpu())
        cur_scores = alpha[0]*cur_scores + alpha[1] * gnn_scores
        neg_valid_preds += [cur_scores]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)
    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        gnn_scores = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        src, dst = pos_test_edge[perm].t().cpu()
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
        cur_scores = torch.sigmoid(model.g_phi(cur_scores).squeeze().cpu())
        cur_scores = alpha[0]*cur_scores + alpha[1] * gnn_scores
        pos_test_preds += [cur_scores]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        gnn_scores = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        src, dst = neg_test_edge[perm].t().cpu()
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
        cur_scores = torch.sigmoid(model.g_phi(cur_scores).squeeze().cpu()) 
        cur_scores = alpha[0]*cur_scores + alpha[1] * gnn_scores
        neg_test_preds += [cur_scores]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    
    del edge_weight
    torch.cuda.empty_cache()
    return results


def main():
    warnings.simplefilter("ignore", UserWarning)
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1024) 
    parser.add_argument('--test_batch_size', type=int, default=1024 * 64) 
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)

    parser.add_argument('--f_edge_dim', type=int, default=8) 
    parser.add_argument('--f_node_dim', type=int, default=128) 
    parser.add_argument('--g_phi_dim', type=int, default=128) 

    parser.add_argument('--gnn', type=str, default='NeoGNN')
    parser.add_argument('--alpha', type=float, default=-1)
    parser.add_argument('--beta', type=float, default=0.1)

    args = parser.parse_args()
    print(args)

    args.dataset = 'collab'
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dataset = PygLinkPropPredDataset(name='ogbl-collab', root='../dataset')
    data = dataset[0]
    edge_index = data.edge_index
    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data = T.ToSparseTensor()(data)
    
    split_edge = dataset.get_edge_split()
    args.num_train_edges = split_edge['train']['edge'].shape[0]
    # Use training + validation edges for inference on test set.
    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t

    data.adj_t = SparseTensor.from_edge_index(edge_index).t()
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    model = NeoGNN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout, args=args).to(device)

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    edge_weight = torch.ones(edge_index.size(1), dtype=float)
    A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                       shape=(data.num_nodes, data.num_nodes))
    A2 = A * A
    A = A + args.beta*A2
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
                        args.batch_size, A, degree, args)

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
                
                valid_performance = results['Hits@50'][1]
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
