if __name__ == '__main__':
    import os

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import pickle
    import copy
    import torch
    import numpy as np
    from tqdm import tqdm
    from torch_geometric import seed_everything
    from torch_geometric.nn import SignedGCN
    from utils.load_old_data import load_edge_index
    from utils.results import save_as_df
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--emb_size', type=int, default=64, help='Embedding dimension for each node.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of SignedGCN layers.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--dataset', type=str, default='Biology', help='The dataset to be used.')
    parser.add_argument('--rounds', type=int, default=1, help='Repeating the training and evaluation process.')

    args = parser.parse_args()
    print(args)

    # init settings
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    torch.use_deterministic_algorithms(True)

    # data information
    with open(os.path.join('datasets', 'processed', args.dataset, 'data_info.pkl'), 'rb') as f:
        data_info = pickle.load(f)

    # SGCN model
    seed_everything(args.seed)
    model = SignedGCN(args.emb_size, args.emb_size, num_layers=args.num_layers, lamb=5).to(device)
    model_state_dict = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    x = torch.randn(size=(data_info['user_num'] + data_info['ques_num'], args.emb_size)).to(device)


    def test(model: SignedGCN, z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor,
             epoch, mode='test') -> dict:
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
            f'{mode}_epoch': epoch,
            f'{mode}_pos_ratio': np.sum(y) / len(y),
            f'{mode}_auc': roc_auc_score(y, pred),
            f'{mode}_f1': f1_score(y, pred) if pred.sum() > 0 else 0,
            f'{mode}_macro_f1': f1_score(y, pred, average='macro') if pred.sum() > 0 else 0,
            f'{mode}_micro_f1': f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
        }
        return res


    def run(round_i: int):
        model.load_state_dict(model_state_dict)

        # load train, val and test set
        g_train, g_val, g_test = (
            load_edge_index(args.dataset, mode='train', round=round_i).to(device),
            load_edge_index(args.dataset, mode='val', round=round_i).to(device),
            load_edge_index(args.dataset, mode='test', round=round_i).to(device)
        )
        train_pos_edge_index, val_pos_edge_index, test_pos_edge_index = \
            g_train[0:2, g_train[2] > 0], g_val[0:2, g_val[2] > 0], g_test[0:2, g_test[2] > 0]
        train_neg_edge_index, val_neg_edge_index, test_neg_edge_index = \
            g_train[0:2, g_train[2] < 0], g_val[0:2, g_val[2] < 0], g_test[0:2, g_test[2] < 0]

        # train the model
        best_res = {'val_auc': 0, 'val_f1': 0}

        # train the model
        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            z = model(x, train_pos_edge_index, train_neg_edge_index)
            loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                z = model(x, train_pos_edge_index, train_neg_edge_index)
            val_res = test(model, z, val_pos_edge_index, val_neg_edge_index, epoch, mode='val')
            if val_res['val_auc'] + val_res['val_f1'] > best_res['val_auc'] + best_res['val_f1'] and epoch >= 100:
                best_res.update(val_res)
                best_res.update(test(model, z, test_pos_edge_index, test_neg_edge_index, epoch, mode='test'))

        print(f'Round {round_i} done.')
        return best_res


    results = [run(i) for i in range(args.rounds)]

    # save the results as a pandas DataFrame
    save_path = os.path.join('results', f'sgcn_{args.dataset}_{args.emb_size}.pkl')
    save_as_df(results, save_path)
