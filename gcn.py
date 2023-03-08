import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse
from torch_geometric.utils import negative_sampling, structured_negative_sampling, coalesce
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

    def create_spectral_features(self, pos_edge_index, neg_edge_index, num_nodes=None):
        r"""Creates :obj:`in_channels` spectral node features based on
        positive and negative edges.

        Args:
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
            num_nodes (int, optional): The number of nodes, *i.e.*
                :obj:`max_val + 1` of :attr:`pos_edge_index` and
                :attr:`neg_edge_index`. (default: :obj:`None`)
        """
        from sklearn.decomposition import TruncatedSVD

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        N = edge_index.max().item() + 1 if num_nodes is None else num_nodes
        edge_index = edge_index.to(torch.device('cpu'))

        pos_val = torch.full((pos_edge_index.size(1),), 2, dtype=torch.float)
        neg_val = torch.full((neg_edge_index.size(1),), 0, dtype=torch.float)
        val = torch.cat([pos_val, neg_val], dim=0)

        row, col = edge_index
        edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        val = torch.cat([val, val], dim=0)

        edge_index, val = coalesce(edge_index, val, num_nodes=N)
        val = val - 1

        # Borrowed from:
        # https://github.com/benedekrozemberczki/SGCN/blob/master/src/utils.py
        edge_index = edge_index.detach().numpy()
        val = val.detach().numpy()
        A = scipy.sparse.coo_matrix((val, edge_index), shape=(N, N))
        svd = TruncatedSVD(n_components=self.em_dim, n_iter=128)
        svd.fit(A)
        x = svd.components_.T
        return torch.from_numpy(x).to(torch.float).to(pos_edge_index.device)

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

    def test(self, z: torch.Tensor, pos_edge_index, neg_edge_index, epoch) -> dict:
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
            'test_epoch': epoch,
            'test_pos_ratio': np.sum(y) / len(y),
            'test_auc': roc_auc_score(y, pred),
            'test_f1': f1_score(y, pred) if pred.sum() > 0 else 0,
            'test_macro_f1': f1_score(y, pred, average='macro') if pred.sum() > 0 else 0,
            'test_micro_f1': f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
        }
        return res


if __name__ == '__main__':
    import os
    from tqdm import tqdm
    from utils.graph_data import GraphData, generate_random_seeds, split_edges
    from utils.results import save_as_df
    from torch_geometric import seed_everything
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--emb_size', type=int, default=64, help='Embedding dimension for each node.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of SignedGCN (implemented by pyg) layers.')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--train_test_ratio', type=float, default=0.2, help='Split the training and test set.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--dataset', type=str, default='Biology', help='The dataset to be used.')
    parser.add_argument('--rounds', type=int, default=1, help='Repeating the training and evaluation process.')

    args = parser.parse_args()
    print(args)

    # init settings
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    dataset_path = os.path.join('datasets', 'PeerWiseData', args.dataset)
    answer_path = os.path.join(dataset_path, 'Answers_CourseX.xlsx')
    question_path = os.path.join(dataset_path, 'Questions_CourseX.xlsx')
    graph_data = GraphData(answer_path, question_path)
    graph_data.summary()  # print some information

    # edge index
    edge_index = torch.from_numpy(graph_data.data.copy().T).to(device)
    pos_edge_index = edge_index[0:2, edge_index[2] > 0]  # positive edge index
    neg_edge_index = edge_index[0:2, edge_index[2] < 0]  # negative edge index

    seed_everything(args.seed)
    model = GCN(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # used for train-test split
    seeds = generate_random_seeds(args.rounds, args.seed)


    def run(round_i: int):
        model.reset_parameters()
        # train-test split
        seed_everything(seeds[round_i])
        train_pos_edge_index, test_pos_edge_index = split_edges(pos_edge_index, args.train_test_ratio)
        train_neg_edge_index, test_neg_edge_index = split_edges(neg_edge_index, args.train_test_ratio)

        # generate user and question embeddings
        # x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index)
        x = torch.randn(size=(graph_data.usr_num + graph_data.qus_num, args.emb_size)).to(device)

        best_res = {'test_auc': 0, 'test_f1': 0}

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
                temp_res.update(model.test(z, test_pos_edge_index, test_neg_edge_index, epoch))
            if temp_res['test_auc'] + temp_res['test_f1'] > best_res['test_auc'] + best_res['test_f1']:
                best_res = temp_res

        print(f'Round {round_i} done.')
        return best_res


    results = [run(i) for i in range(args.rounds)]

    # save the results as a pandas DataFrame
    save_path = os.path.join('results', 'gcn_baseline_' + args.dataset + '.pkl')
    save_as_df(results, save_path)
