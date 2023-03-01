import sys
import os
import numpy as np
import torch
from src.signed_graph_model.model import SGNNEnc

# TODO: include NLP model
sys.path.append('src/pre_train_model')
from src.pre_train_model.embedding import load_dataset
from src.pre_train_model.nlp_model import NLPModel
from sklearn.metrics import f1_score, roc_auc_score
from utils.graph_data import GraphData, create_perspectives, train_test_split, perturb_structure, generate_random_seeds
from torch_geometric import seed_everything
from utils.results import save_as_df
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
parser.add_argument('--emb_size', type=int, default=128, help='Embedding dimension for each node.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of SignedGCN (implemented by pyg) layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout parameter.')
parser.add_argument('--linear_predictor_layers', type=int, default=1,
                    help='Number of MLP layers (0-4) to make prediction from learned embeddings.')
parser.add_argument('--mask_ratio', type=float, default=0.1, help='Random mask ratio')
parser.add_argument('--beta', type=float, default=5e-4, help='Control contribution of loss contrastive.')
parser.add_argument('--alpha', type=float, default=0.8, help='Control the contribution of inter and intra loss.')
parser.add_argument('--tau', type=float, default=0.05, help='Temperature parameter.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
# TODO: add NLP learning rate
parser.add_argument('--nlp_lr', type=float, default=0.001, help='Initial learning rate for the pretrained NLP model.')
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

# TODO: add question data
qus_x = load_dataset(question_path).astype(str)  # np.ndarray
qus_text_arr = [' '.join(row) for row in qus_x]  # string array

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
    p1, p2u, p2q = create_perspectives(trn_data)

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

    # TODO: only random embed users
    # generate user embedding
    usr_emb = torch.rand(generator=torch.manual_seed(args.seed),
                         size=(graph_data.usr_num, args.emb_size)).to(device)

    # build model
    seed_everything(args.seed)
    model = SGNNEnc(graph_data.usr_num, graph_data.qus_num, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TODO: ADD NLP model & optimizer
    nlp_model = NLPModel(max_length=512, feature_size=args.emb_size, device=device).to(device)
    nlp_optimizer = torch.optim.Adam(nlp_model.parameters(), lr=args.nlp_lr)

    # train model
    best_res = {'test_auc': 0, 'test_f1': 0}
    # TODO: save result from each epoch
    temp_res_list = []

    for epoch in range(args.epochs):
        model.train()
        # TODO: training the NLP model
        nlp_model.train()
        # TODO: question embeddings using NLP
        qus_emb = torch.cat([nlp_model(text) for text in qus_text_arr], dim=0)
        # TODO: new embedding using random user embedding and NLP question embedding
        x = torch.cat([usr_emb, qus_emb], dim=0)  # user embedding first

        embeddings = model(x, edges, masks)
        loss_contrastive = model.compute_contrastive_loss(embeddings)  # contrastive loss
        y_score = model.predict_edges(model.emb_out, uid_trn, qid_trn)  # label loss
        loss_label = model.compute_label_loss(y_score, trn_y)
        loss = args.beta * loss_contrastive + loss_label

        # TODO: the NLP loss
        optimizer.zero_grad()
        nlp_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        nlp_optimizer.step()

        # model evaluation
        model.eval()
        nlp_model.eval()

        temp_res = {}
        with torch.no_grad():
            y_score_trn = model.predict_edges(model.emb_out, uid_trn, qid_trn)
            y_score_tst = model.predict_edges(model.emb_out, uid_tst, qid_tst)
            temp_res.update(test_and_val(y_score_trn, trn_y, mode='train', epoch=epoch))
            temp_res.update(test_and_val(y_score_tst, tst_y, mode='test', epoch=epoch))
        if temp_res['test_auc'] + temp_res['test_f1'] > best_res['test_auc'] + best_res['test_f1']:
            best_res = temp_res

        # TODO: save model parameters & results every 10 epochs
        if epoch % 10 == 0:
            torch.save(model, os.path.join('save_models', 'SGCN_model.pth'))
            torch.save(nlp_model, os.path.join('save_models', 'NLP_model.pth'))

        # TODO: save result after each epoch
        temp_res_list.append(temp_res)
        save_as_df(temp_res_list, os.path.join('results', f'NLP_epoch_x.pkl'), show=False)

    print(f'Round {round_i} done.')
    return best_res


if __name__ == '__main__':
    results = []

    for i in range(args.rounds):
        results.append(run(i))

    # save the results as a pandas DataFrame
    save_path = os.path.join('results', 'NLP_' + args.dataset + '.pkl')
    save_as_df(results, save_path)
