import os
import numpy as np
import torch
from src.signed_graph_model.model import SGNNEnc
from sklearn.metrics import f1_score, roc_auc_score
from utils.graph_data import GraphData, create_perspectives, train_test_split, perturb_structure, generate_random_seeds
from torch_geometric import seed_everything
from utils.results import save_as_df
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
parser.add_argument('--emb_size', type=int, default=128, help='Embedding dimension for each node.')
parser.add_argument('--num_layers', type=int, default=1, help='Number of SignedGCN (implemented by pyg) layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout parameter.')
parser.add_argument('--linear_predictor_layers', type=int, default=1,
                    help='Number of MLP layers (0-4) to make prediction from learned embeddings.')
parser.add_argument('--mask_ratio', type=float, default=0.1, help='Random mask ratio')
parser.add_argument('--beta', type=float, default=5e-4, help='Control contribution of loss contrastive.')
parser.add_argument('--alpha', type=float, default=0.8, help='Control the contribution of inter and intra loss.')
parser.add_argument('--tau', type=float, default=0.05, help='Temperature parameter.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--train_test_ratio', type=float, default=0.2, help='Split the training and test set.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
parser.add_argument('--dataset', type=str, default='Biology', help='The dataset to be used.')
parser.add_argument('--rounds', type=int, default=1, help='Repeating the training and evaluation process.')
parser.add_argument('--usr_pos_thres', type=int, default=2,
                    help='The positive threshold to link edges between users in perspective 2.')
parser.add_argument('--usr_neg_thres', type=int, default=-2,
                    help='The negative threshold to link edges between users in perspective 2. (negative value)')
parser.add_argument('--qus_pos_thres', type=int, default=5,
                    help='The positive threshold to link edges between questions in perspective 2.')
parser.add_argument('--qus_neg_thres', type=int, default=-5,
                    help='The negative threshold to link edges between questions in perspective 2. (negative value)')

args = parser.parse_args()
print(args)

# init settings
device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
dataset_path = os.path.join('datasets', 'PeerWiseData', args.dataset)
answer_path = os.path.join(dataset_path, 'Answers_CourseX.xlsx')
question_path = os.path.join(dataset_path, 'Questions_CourseX.xlsx')
graph_data = GraphData(answer_path, question_path)
graph_data.summary()  # print some information

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
    seed_everything(args.seed)

    # train-test split
    trn_data, tst_data = train_test_split(graph_data.data, args.train_test_ratio, seed=seeds[round_i])

    # create perspective 1 and perspective 2 (user-question, user-user, question-question)
    p1, p2u, p2q = create_perspectives(trn_data, args)

    # data augmentation
    # Graph 1
    g1_uq = perturb_structure(p1, augment='flip', args=args)
    mask_g1 = (g1_uq[2] == 1).to(device)
    edge_index_g1 = g1_uq[0:2].to(device)
    # Graph 2
    g2_uq = perturb_structure(p1, augment='delete', args=args)
    mask_g2 = (g2_uq[2] == 1).to(device)
    edge_index_g2 = g2_uq[0:2].to(device)
    # Graph 3
    g3_u = perturb_structure(p2u, augment='flip', args=args)
    g3_q = perturb_structure(p2q, augment='flip', args=args)
    mask_g3_u = (g3_u[2] == 1).to(device)
    mask_g3_q = (g3_q[2] == 1).to(device)
    edge_index_g3_u = g3_u[0:2].to(device)
    edge_index_g3_q = g3_q[0:2].to(device)
    # Graph 4
    g4_u = perturb_structure(p2u, augment='delete', args=args)
    g4_q = perturb_structure(p2q, augment='delete', args=args)
    mask_g4_u = (g4_u[2] == 1).to(device)
    mask_g4_q = (g4_q[2] == 1).to(device)
    edge_index_g4_u = g4_u[0:2].to(device)
    edge_index_g4_q = g4_q[0:2].to(device)

    # user id
    uid_trn = torch.from_numpy(trn_data[:, 0]).long().to(device)
    uid_tst = torch.from_numpy(tst_data[:, 0]).long().to(device)
    # question id
    qid_trn = torch.from_numpy(trn_data[:, 1]).long().to(device)
    qid_tst = torch.from_numpy(tst_data[:, 1]).long().to(device)
    # labels
    trn_y = torch.from_numpy(trn_data[:, 2] == 1).float().to(device)
    tst_y = torch.from_numpy(tst_data[:, 2] == 1).float().to(device)

    # edge index
    edges = edge_index_g1, edge_index_g2, edge_index_g3_u, edge_index_g3_q, edge_index_g4_u, edge_index_g4_q
    # positive, negative edge mask
    masks = mask_g1, mask_g2, mask_g3_u, mask_g3_q, mask_g4_u, mask_g4_q
    # generate random embeddings
    x = torch.rand(generator=torch.manual_seed(args.seed),
                   size=(graph_data.usr_num + graph_data.qus_num, args.emb_size)).to(device)

    # build model
    seed_everything(args.seed)
    model = SGNNEnc(graph_data.usr_num, graph_data.qus_num, args.emb_size, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train model
    best_res = {'test_auc': 0, 'test_f1': 0}

    for epoch in range(args.epochs):
        model.train()
        embeddings = model(x, edges, masks)
        loss_contrastive = model.compute_contrastive_loss(embeddings)  # contrastive loss
        y_score = model.predict_edges(model.emb_out, uid_trn, qid_trn)  # label loss
        loss_label = model.compute_label_loss(y_score, trn_y)
        loss = args.beta * loss_contrastive + loss_label

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        temp_res = {}
        with torch.no_grad():
            y_score_trn = model.predict_edges(model.emb_out, uid_trn, qid_trn)
            y_score_tst = model.predict_edges(model.emb_out, uid_tst, qid_tst)
            temp_res.update(test_and_val(y_score_trn, trn_y, mode='train', epoch=epoch))
            temp_res.update(test_and_val(y_score_tst, tst_y, mode='test', epoch=epoch))
        if temp_res['test_auc'] + temp_res['test_f1'] > best_res['test_auc'] + best_res['test_f1']:
            best_res = temp_res

    print(f'Round {round_i} done.')
    return best_res


if __name__ == '__main__':
    results = []

    for i in range(args.rounds):
        results.append(run(i))

    # save the results as a pandas DataFrame
    save_path = os.path.join('results', 'BaseLine_' + args.dataset + '.pkl')
    save_as_df(results, save_path)
