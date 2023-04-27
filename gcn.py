import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import negative_sampling, structured_negative_sampling
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, args, lamb=5):
        super(GCN, self).__init__()
        self.em_dim = args.emb_size
        self.num_layers = args.num_layers
        self.lamb = lamb

        self.lin = torch.nn.Linear(2 * self.em_dim, 3)

        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=self.em_dim, out_channels=self.em_dim) for _ in range(self.num_layers)])
        self.reset_parameters()

    def forward(self, x, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        z = self.convs[0](x, edge_index)
        for i in range(1, self.num_layers):
            z = self.convs[i](z, edge_index)
        return z

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def discriminate(self, z, edge_index):
        """
        Given node embedding z, classified the link relation between node pairs
        :param z: node features
        :param edge_index: edge indicies
        :return:
        """
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)

    def nll_loss(self, z, pos_edge_index, neg_edge_index):
        """Computes the discriminator loss based on node embeddings :obj:`z`,
        and positive edges :obj:`pos_edge_index` and negative edges
        :obj:`neg_edge_index`.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += F.nll_loss(
            self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.size(1),), 0))
        nll_loss += F.nll_loss(
            self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.size(1),), 1))
        nll_loss += F.nll_loss(
            self.discriminate(z, none_edge_index),
            none_edge_index.new_full((none_edge_index.size(1),), 2))
        return nll_loss / 3.0

    def pos_embedding_loss(self, z, pos_edge_index):
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z, neg_edge_index):
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def loss(self, z, pos_edge_index, neg_edge_index):
        """Computes the overall objective.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + self.lamb * (loss_1 + loss_2)

    def test(self, z: torch.Tensor, pos_edge_index, neg_edge_index, epoch, mode='test') -> dict:
        """Evaluates node embeddings :obj:`z` on positive and negative test
        edges by computing AUC and F1 scores.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        from sklearn.metrics import f1_score, roc_auc_score

        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]
        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat(
            [pred.new_ones((pos_p.size(0))),
             pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        res = {
            f'{mode}_epoch': epoch,
            f'{mode}_pos_ratio': np.sum(y) / len(y),
            f'{mode}_auc': roc_auc_score(y, pred),
            f'{mode}_f1': f1_score(y, pred) if pred.sum() > 0 else 0,
            f'{mode}_macro_f1': f1_score(y, pred, average='macro') if pred.sum() > 0 else 0,
            f'{mode}_micro_f1': f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
        }
        return res


if __name__ == '__main__':
    import os
    import pickle
    import copy
    from tqdm import tqdm
    from utils.results import save_as_df
    from torch_geometric import seed_everything
    from utils.load_old_data import load_edge_index
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--emb_size', type=int, default=32, help='Embedding dimension for each node.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of GNN layers.')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--dataset', type=str, default='Biology', help='The dataset to be used.')
    parser.add_argument('--rounds', type=int, default=2, help='Repeating the training and evaluation process.')

    args = parser.parse_args()
    print(args)

    # init settings
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'

    # data information
    with open(os.path.join('datasets', 'processed', args.dataset, 'data_info.pkl'), 'rb') as f:
        data_info = pickle.load(f)

    seed_everything(args.seed)
    model = GCN(args).to(device)
    model_state_dict = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    x = torch.randn(size=(data_info['user_num'] + data_info['ques_num'], args.emb_size)).to(device)  # random embeddings


    def run(round_i: int):
        model.load_state_dict(model_state_dict)

        # load train-test dataset
        g_train = load_edge_index(args.dataset, train=True, round=round_i).to(device)
        g_test = load_edge_index(args.dataset, train=False, round=round_i).to(device)

        train_pos_edge_index, train_neg_edge_index = g_train[0:2, g_train[2] > 0], g_train[0:2, g_train[2] < 0]
        test_pos_edge_index, test_neg_edge_index = g_test[0:2, g_test[2] > 0], g_test[0:2, g_test[2] < 0]

        # train the model
        best_res = {'train_auc': 0, 'train_f1': 0}

        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            z = model(x, train_pos_edge_index, train_neg_edge_index)
            loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
            loss.backward()
            optimizer.step()

            # evaluate the model
            model.eval()
            temp_res = {}
            with torch.no_grad():
                z = model(x, train_pos_edge_index, train_neg_edge_index)
                temp_res.update(model.test(z, train_pos_edge_index, train_neg_edge_index, epoch, mode='train'))
                temp_res.update(model.test(z, test_pos_edge_index, test_neg_edge_index, epoch, mode='test'))
            if temp_res['train_auc'] + temp_res['train_f1'] > best_res['train_auc'] + best_res['train_f1']:
                best_res = temp_res

        print(f'Round {round_i} done.')
        return best_res


    results = [run(i) for i in range(args.rounds)]

    # save the results as a pandas DataFrame
    save_path = os.path.join('results', f'gcn_baseline_{args.dataset}.pkl')
    save_as_df(results, save_path)
