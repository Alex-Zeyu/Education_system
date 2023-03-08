if __name__ == '__main__':
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from torch_geometric import seed_everything
    from torch_geometric.nn import SignedGCN
    from utils.graph_data import GraphData, generate_random_seeds
    from utils.results import save_as_df
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--emb_size', type=int, default=128, help='Embedding dimension for each node.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of SignedGCN (implemented by pyg) layers.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--train_test_ratio', type=float, default=0.2, help='Split the training and test set.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
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

    # SGCN model
    seed_everything(args.seed)
    model = SignedGCN(args.emb_size, args.emb_size, num_layers=args.num_layers, lamb=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # used for train-test split
    seeds = generate_random_seeds(args.rounds, args.seed)


    def test(model: SignedGCN, z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor,
             epoch) -> dict:
        """Evaluates node embeddings :obj:`z` on positive and negative test
        edges by computing AUC and F1 scores.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        from sklearn.metrics import f1_score, roc_auc_score

        with torch.no_grad():
            pos_p = model.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = model.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]
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


    def run(round_i: int):
        model.reset_parameters()

        # train-test split
        seed_everything(seeds[round_i])
        train_pos_edge_index, test_pos_edge_index = model.split_edges(pos_edge_index, args.train_test_ratio)
        train_neg_edge_index, test_neg_edge_index = model.split_edges(neg_edge_index, args.train_test_ratio)

        # user and question embeddings (can't learn stuff)
        # x = torch.rand(generator=torch.manual_seed(args.seed),
        #                size=(graph_data.usr_num + graph_data.qus_num, args.emb_size)).to(device)
        x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index)

        best_res = {'test_auc': 0, 'test_f1': 0}

        # train the model
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
                temp_res.update(test(model, z, test_pos_edge_index, test_neg_edge_index, epoch))
            if temp_res['test_auc'] + temp_res['test_f1'] > best_res['test_auc'] + best_res['test_f1']:
                best_res = temp_res

        print(f'Round {round_i} done.')
        return best_res


    results = [run(i) for i in range(args.rounds)]

    # save the results as a pandas DataFrame
    save_path = os.path.join('results', 'sgcn_baseline_' + args.dataset + '.pkl')
    save_as_df(results, save_path)
