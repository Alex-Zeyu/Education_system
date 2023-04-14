if __name__ == '__main__':
    import os
    import copy
    import torch
    import numpy as np
    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score, f1_score
    from torch_geometric import seed_everything
    from src.signed_graph_model.model import GAT_CL
    from utils.results import save_as_df
    from utils.load_old_data import load_edge_index
    import pickle
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--emb_size', type=int, default=64, help='Embedding dimension for each node.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of GNN (implemented by pyg) layers.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout parameter.')
    parser.add_argument('--linear_predictor_layers', type=int, default=1,
                        help='Number of MLP layers (0-4) to make prediction from learned embeddings.')
    parser.add_argument('--mask_ratio', type=float, default=0.1, help='Random mask ratio')
    parser.add_argument('--beta', type=float, default=5e-4, help='Control contribution of loss contrastive.')
    parser.add_argument('--alpha', type=float, default=0.8, help='Control the contribution of inter and intra loss.')
    parser.add_argument('--tau', type=float, default=0.05, help='Temperature parameter.')
    parser.add_argument('--lr', type=float, default=5e-3, help='Initial learning rate.')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Split the training and test set.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--early_stop_steps', type=int, default=10, help='Early stopping.')
    parser.add_argument('--dataset', type=str, default='Biology', help='The dataset to be used.')
    parser.add_argument('--rounds', type=int, default=2, help='Repeating the training and evaluation process.')
    # NLP settings
    parser.add_argument('--nlp_method', type=str, default='glove', help='NLP embedding method.')
    parser.add_argument('--nlp_lr', type=float, default=1e-4, help='Initial NLP learning rate.')

    args = parser.parse_args()
    print(args)

    # init settings
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'

    # data information
    with open(os.path.join('datasets', 'processed', args.dataset, 'data_info.pkl'), 'rb') as f:
        data_info = pickle.load(f)

    # GAT contrastive model
    seed_everything(args.seed)
    model = GAT_CL(args, device).to(device)
    model_state_dict = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # question embedding
    if args.nlp_method == 'glove':
        nlp_emb_size = 100
    elif args.nlp_method == 'roberta':
        nlp_emb_size = 768
    else:
        raise Exception('Invalid NLP embedding method.')
    x_user = torch.randn(size=(data_info['user_num'], args.emb_size)).to(device)  # user random embeddings
    x_ques_raw = torch.load(os.path.join('embedding', args.dataset, f'{args.nlp_method}.pt')).to(device)

    # transform the NLP output
    nlp_transform = torch.nn.Linear(nlp_emb_size, args.emb_size, bias=False).to(device)
    nlp_state_dict = copy.deepcopy(nlp_transform.state_dict())
    nlp_optimizer = torch.optim.Adam(nlp_transform.parameters(), lr=args.nlp_lr)

    @torch.no_grad()
    def test_and_val(y_score, y, mode='test', epoch=0):
        y_score = y_score.cpu().numpy()
        y = y.cpu().numpy()
        y_pred = np.where(y_score >= .5, 1, 0)
        res = {
            f'{mode}_epoch': epoch,
            f'{mode}_pos_ratio': np.sum(y) / len(y),
            f'{mode}_auc': roc_auc_score(y, y_pred),
            f'{mode}_f1': f1_score(y, y_pred),
            f'{mode}_macro_f1': f1_score(y, y_pred, average='macro'),
            f'{mode}_micro_f1': f1_score(y, y_pred, average='micro')
        }
        return res


    def run(round_i: int):
        model.load_state_dict(model_state_dict)  # reset parameters
        nlp_transform.load_state_dict(nlp_state_dict)

        # load train-test dataset
        g_train = load_edge_index(args.dataset, train=True, round=round_i).to(device)
        g_test = load_edge_index(args.dataset, train=False, round=round_i).to(device)

        # graph augmentation
        # generate augmentation mask
        mask1 = torch.ones(g_train.size(1), dtype=torch.bool)
        mask1[torch.randperm(mask1.size(0))[:int(args.mask_ratio * mask1.size(0))]] = 0

        mask2 = torch.ones(g_train.size(1), dtype=torch.bool)
        mask2[torch.randperm(mask2.size(0))[:int(args.mask_ratio * mask2.size(0))]] = 0

        g1 = g_train[:, mask1]  # delete edges
        g2 = g_train.detach().clone()  # flip signs
        g2[2, ~mask2] *= -1

        edge_index_g1_pos = g1[0:2, g1[2] > 0]  # graph 1
        edge_index_g1_neg = g1[0:2, g1[2] < 0]

        edge_index_g2_pos = g2[0:2, g2[2] > 0]  # graph 2
        edge_index_g2_neg = g2[0:2, g2[2] < 0]

        # train the model
        best_res = {'train_auc': 0, 'train_f1': 0}
        best_loss, early_stop_cnt = np.Inf, 0

        for epoch in tqdm(range(args.epochs)):
            model.train()
            nlp_transform.train()
            x = torch.cat([x_user, nlp_transform(x_ques_raw)], dim=0)
            emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg = model(x, edge_index_g1_pos, edge_index_g2_pos,
                                                                   edge_index_g1_neg, edge_index_g2_neg)
            # contrastive loss
            loss_contrastive = model.compute_contrastive_loss(emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg)
            y_score = model.predict_edges(model.norm_embs, g_train[0], g_train[1])
            loss_label = model.compute_label_loss(y_score, (g_train[2] == 1).float())
            loss = args.beta * loss_contrastive + loss_label

            # early stopping
            if loss < best_loss:
                best_loss = loss
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            if early_stop_cnt >= args.early_stop_steps:
                break

            optimizer.zero_grad()
            nlp_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nlp_optimizer.step()

            # model evaluation
            model.eval()
            nlp_transform.eval()
            temp_res = {}
            with torch.no_grad():
                x = torch.cat([x_user, nlp_transform(x_ques_raw)], dim=0)
                z = model(x, edge_index_g1_pos, edge_index_g2_pos, edge_index_g1_neg, edge_index_g2_neg)
                z = model.linear_combine(torch.cat(z, dim=-1))
                z = torch.nn.functional.normalize(z, p=2, dim=1)

                y_score_train = model.predict_edges(z, g_train[0], g_train[1])
                y_score_test = model.predict_edges(z, g_test[0], g_test[1])
                temp_res.update(test_and_val(y_score_train, (g_train[2] == 1).float(), mode='train', epoch=epoch))
                temp_res.update(test_and_val(y_score_test, (g_test[2] == 1).float(), mode='test', epoch=epoch))
            if temp_res['train_auc'] + temp_res['train_f1'] > best_res['train_auc'] + best_res['train_f1']:
                best_res = temp_res
        print(f'Round {round_i} done.')
        return best_res


    results = [run(i) for i in range(args.rounds)]
    # save the results as a pandas DataFrame
    save_path = os.path.join('results', f'gat_cl_{args.nlp_method}' + args.dataset + '.pkl')
    save_as_df(results, save_path)
