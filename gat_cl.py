if __name__ == '__main__':
    import os

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import copy
    import torch
    from tqdm import tqdm
    from torch_geometric import seed_everything
    from src.signed_graph_model.model import GAT_CL
    from utils.results import test_and_val, save_as_df
    from utils.load_old_data import load_edge_index
    import pickle
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--emb_size', type=int, default=64, help='Embedding dimension for each node.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN (implemented by pyg) layers.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout parameter.')
    parser.add_argument('--linear_predictor_layers', type=int, default=1, choices=range(5),
                        help='Number of MLP layers (0-4) to make prediction from learned embeddings.')
    parser.add_argument('--mask_ratio', type=float, default=0.1, help='Random mask ratio')
    parser.add_argument('--beta', type=float, default=5e-4, help='Control contribution of loss contrastive.')
    parser.add_argument('--alpha', type=float, default=0.8, help='Control the contribution of inter and intra loss.')
    parser.add_argument('--tau', type=float, default=0.05, help='Temperature parameter.')
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

    # GAT contrastive model
    seed_everything(args.seed)
    model = GAT_CL(args).to(device)
    model_state_dict = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    x = torch.randn(size=(data_info['user_num'] + data_info['ques_num'], args.emb_size)).to(device)


    def run(round_i: int):
        model.load_state_dict(model_state_dict)  # reset parameters

        # load train, val and test set
        g_train = load_edge_index(args.dataset, mode='train', round=round_i).to(device)
        g_val = load_edge_index(args.dataset, mode='val', round=round_i).to(device)
        g_test = load_edge_index(args.dataset, mode='test', round=round_i).to(device)
        # x = model.create_spectral_features(g_train[0:2, g_train[2] > 0], g_train[0:2, g_train[2] < 0])
        y_label_train, y_label_val, y_label_test = (g_train[2] == 1).float(), (g_val[2] == 1).float(), (
                    g_test[2] == 1).float()  # y-labels

        # graph augmentation
        # generate augmentation mask
        mask1 = torch.ones(g_train.size(1), dtype=torch.bool)
        mask2 = torch.ones(g_train.size(1), dtype=torch.bool)
        mask1[torch.randperm(mask1.size(0))[:int(args.mask_ratio * mask1.size(0))]] = 0
        mask2[torch.randperm(mask2.size(0))[:int(args.mask_ratio * mask2.size(0))]] = 0
        g1, g2 = g_train.clone(), g_train.clone()  # flip sign
        g1[2, ~mask1] *= -1
        g2[2, ~mask2] *= -1

        edge_index_g1_pos = g1[0:2, g1[2] > 0]  # graph 1
        edge_index_g1_neg = g1[0:2, g1[2] < 0]
        edge_index_g2_pos = g2[0:2, g2[2] > 0]  # graph 2
        edge_index_g2_neg = g2[0:2, g2[2] < 0]

        # train the model
        best_res = {'val_auc': 0, 'val_macro_f1': 0}

        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg = model(x, edge_index_g1_pos, edge_index_g2_pos,
                                                                   edge_index_g1_neg, edge_index_g2_neg)
            # contrastive loss
            loss_contrastive = model.compute_contrastive_loss(emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg)
            y_score = model.predict_edges(model.embs, g_train[0], g_train[1])
            loss_label = model.compute_label_loss(y_score, y_label_train)
            loss = args.beta * loss_contrastive + loss_label
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                z = model(x, edge_index_g1_pos, edge_index_g2_pos, edge_index_g1_neg, edge_index_g2_neg)
                z = model.linear_combine(torch.cat(z, dim=-1))
                # z = torch.nn.functional.normalize(z, p=2, dim=1)
                y_score_val = model.predict_edges(z, g_val[0], g_val[1])
            val_res = test_and_val(y_score_val, y_label_val, mode='val', epoch=epoch)

            if val_res['val_auc'] + val_res['val_macro_f1'] > best_res['val_auc'] + best_res['val_macro_f1']:
                best_res.update(val_res)
                y_score_test = model.predict_edges(z, g_test[0], g_test[1])
                best_res.update(test_and_val(y_score_test, y_label_test, mode='test', epoch=epoch))

        print(f'Round {round_i} done.')
        return best_res


    results = [run(i) for i in range(args.rounds)]
    # save the results as a pandas DataFrame
    save_path = os.path.join('results', f'gat_cl_{args.dataset}_{args.emb_size}.pkl')
    save_as_df(results, save_path)
