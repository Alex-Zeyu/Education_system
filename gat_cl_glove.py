if __name__ == '__main__':
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score, f1_score
    from torch_geometric import seed_everything
    from src.signed_graph_model.model import GAT_CL
    from utils.graph_data import GraphData, generate_random_seeds, split_edges_undirected
    from utils.results import save_as_df
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
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--nlp_lr', type=float, default=1e-3, help='Initial NLP learning rate.')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Split the training and test set.')
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
    print(graph_data)  # print some information

    # edge index with sign
    g = graph_data.get_undirected_edge_index_with_sign().to(device)

    # GAT contrastive model
    seed_everything(args.seed)
    model = GAT_CL(args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # user random embeddings
    usr_emb = torch.randn(size=(graph_data.usr_num, args.emb_size)).to(device)
    # load NLP embedding for questions
    nlp_emb_path = os.path.join('embedding', f'{args.dataset}_glove.pt')
    if os.path.exists(nlp_emb_path):
        qus_emb_raw = torch.load(nlp_emb_path)
    else:
        import sys
        sys.path.append('src/pre_train_model')
        from src.pre_train_model.embedding import load_dataset, clean_html
        from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
        from flair.data import Sentence

        qus_x = load_dataset(graph_data.get_question_df(), None).astype(str)  # np.ndarray
        qus_text_arr = [clean_html(' '.join(row)) for row in qus_x]  # string array
        qus_sent_arr = [Sentence(s) for s in qus_text_arr]  # Sentence array
        # use pretrained NLP model
        nlp_model = DocumentPoolEmbeddings([WordEmbeddings('glove')], fine_tune_mode='none')
        for s in qus_sent_arr:
            nlp_model.embed(s)
        qus_emb_raw = torch.cat([s.embedding.unsqueeze(0) for s in qus_sent_arr], dim=0)  # raw question embeddings
        torch.save(qus_emb_raw, nlp_emb_path)

    # transform the NLP output
    nlp_emb_size = 100
    nlp_transform = torch.nn.Linear(nlp_emb_size, args.emb_size, bias=False).to(device)
    nlp_trans_optimizer = torch.optim.Adam(nlp_transform.parameters(), lr=args.nlp_lr, weight_decay=5e-4)

    # used for train-test split
    seeds = generate_random_seeds(args.rounds, args.seed)


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
        model.reset_parameters()
        nlp_transform.reset_parameters()

        # train-test split
        seed_everything(seeds[round_i])
        # edge index with signs
        g_train, g_test = split_edges_undirected(g, args.test_ratio)

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
        best_res = {'test_auc': 0, 'test_f1': 0}

        for epoch in tqdm(range(args.epochs)):
            model.train()
            nlp_transform.train()

            qus_emb = nlp_transform(qus_emb_raw)
            qus_emb = torch.relu(qus_emb)
            x = torch.cat([usr_emb, qus_emb], dim=0)
            emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg = model(x, edge_index_g1_pos, edge_index_g2_pos,
                                                                   edge_index_g1_neg, edge_index_g2_neg)
            # contrastive loss
            loss_contrastive = model.compute_contrastive_loss(emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg)
            y_score = model.predict_edges(model.norm_embs, g_train[0], g_train[1])
            loss_label = model.compute_label_loss(y_score, (g_train[2] == 1).float())

            loss = args.beta * loss_contrastive + loss_label

            optimizer.zero_grad()
            nlp_trans_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nlp_trans_optimizer.step()

            # model evaluation
            model.eval()
            nlp_transform.eval()
            temp_res = {}
            with torch.no_grad():
                z = model(x, edge_index_g1_pos, edge_index_g2_pos, edge_index_g1_neg, edge_index_g2_neg)
                z = [torch.nn.functional.normalize(emb, p=2, dim=1) for emb in z]
                z = model.linear_combine(torch.cat(z, dim=-1))
                z = torch.nn.functional.normalize(z, p=2, dim=1)

                y_score_test = model.predict_edges(z, g_test[0], g_test[1])
                temp_res.update(test_and_val(y_score_test, (g_test[2] == 1).float(), mode='test', epoch=epoch))
            if temp_res['test_auc'] + temp_res['test_f1'] > best_res['test_auc'] + best_res['test_f1']:
                best_res = temp_res

        print(f'Round {round_i} done.')
        return best_res


    results = [run(i) for i in range(args.rounds)]
    # save the results as a pandas DataFrame
    save_path = os.path.join('results', 'gat_cl_glove_' + args.dataset + '.pkl')
    save_as_df(results, save_path)
