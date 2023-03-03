import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import negative_sampling
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


class GraphData:
    def __init__(self, ans_path: str, qus_path: str) -> None:
        """Store the xlsx data into numpy array, with format [user_id, question_id, sign]"""
        # load data
        self.ans = pd.read_excel(ans_path, sheet_name=0)
        self.qus = pd.read_excel(qus_path, sheet_name=0)

        # create sign
        data = self.ans[['QuestionID', 'UserID', 'Answer']].merge(self.qus[['QuestionID', 'Answer']], on='QuestionID')
        data['Sign'] = np.where(data.Answer_x == data.Answer_y, 1, -1)
        self.data = data.drop(columns=['Answer_x', 'Answer_y'], inplace=False)

        # count user and question number
        self.qus_num = self.data['QuestionID'].nunique()
        self.usr_num = self.data['UserID'].nunique()

        # encode users and questions
        u_le = LabelEncoder()  # user encoder
        q_le = LabelEncoder()  # question encoder
        self.data['UserID'] = u_le.fit_transform(self.data['UserID'])
        self.data['QuestionID'] = q_le.fit_transform(self.data['QuestionID'])

        self.data = self.data.reindex(columns=['UserID', 'QuestionID', 'Sign'])

        # map encode id -> actual id
        # self.enc_usr_map = u_le.classes_
        # self.enc_qus_map = q_le.classes_

        self.data = self.data.to_numpy()
        self.data[:, 1] += self.usr_num  # add offset to question id

    def summary(self):
        """Print some information about the data"""
        print('\nData Summary:')
        print(f'number of edges: {len(self.data)}')
        print(f'number of users: {self.usr_num}')
        print(f'number of questions: {self.qus_num}\n')


def create_perspectives(arr: np.ndarray, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return perspective 1 (inter): edge index user->question &
    perspective 2 (intra): edge index user->user, question->question
    """
    # create perspective 1
    perspective_1 = arr.copy().T

    # create perspective 2
    adj_list_uq_pos, adj_list_uq_neg = defaultdict(list), defaultdict(list)
    adj_list_qu_pos, adj_list_qu_neg = defaultdict(list), defaultdict(list)
    for u, q, s in arr:
        if s == 1:
            adj_list_uq_pos[u].append(q)
            adj_list_qu_pos[q].append(u)
        elif s == -1:
            adj_list_uq_neg[u].append(q)
            adj_list_qu_neg[q].append(u)
        else:
            raise Exception("Edge sign must be +/-1.")

    adj_list_u_u = defaultdict(lambda: defaultdict(int))
    adj_list_q_q = defaultdict(lambda: defaultdict(int))

    for u, q, sign in arr:
        for q2 in adj_list_uq_pos[u]:
            adj_list_q_q[q][q2] += sign
        for q2 in adj_list_uq_neg[u]:
            adj_list_q_q[q][q2] -= sign
        for u2 in adj_list_qu_pos[q]:
            adj_list_u_u[u][u2] += sign
        for u2 in adj_list_qu_neg[q]:
            adj_list_u_u[u][u2] -= sign

    perspective_2_u = []
    perspective_2_q = []

    for u in adj_list_u_u.keys():
        for u2, val in adj_list_u_u[u].items():
            if u == u2:
                continue
            if val >= args.usr_pos_thres:
                perspective_2_u.append([u, u2, 1])
            elif val <= args.usr_neg_thres:  # should be a negative number
                perspective_2_u.append([u, u2, -1])

    for q in adj_list_q_q.keys():
        for q2, val in adj_list_q_q[q].items():
            if q == q2:
                continue
            if val >= args.qus_pos_thres:
                perspective_2_q.append([q, q2, 1])
            elif val <= args.qus_neg_thres:  # should be a negative number
                perspective_2_q.append([q, q2, -1])

    return perspective_1, np.array(perspective_2_u).T, np.array(perspective_2_q).T


def generate_random_seeds(n: int, seed: int) -> np.ndarray:
    """Generate seed, can be used to split the training and test set"""
    g = np.random.default_rng(seed)
    return g.integers(low=0, high=1e4, size=n)


def generate_1d_mask(n, mask_ratio: float) -> np.ndarray:
    """Generate 1d mask boolean array"""
    return np.random.binomial(1, mask_ratio, n).astype(bool)


def train_test_split(arr: np.ndarray, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the edge index array with shape (n_edges, 3) into training and test sets"""
    mask = np.random.default_rng(seed).binomial(1, test_ratio, arr.shape[0]).astype(bool)
    trn_arr = arr[~mask, :]
    tst_arr = arr[mask, :]
    return trn_arr, tst_arr


def perturb_structure(perspective: np.ndarray, augment: str, args) -> torch.Tensor:
    """Perturb the edge index array with label, perspective: np.array with shape (3, n_edges)"""
    new_edge_index = perspective.copy()  # copy of the original edge index
    if augment == 'flip':
        mask = generate_1d_mask(perspective.shape[1], args.mask_ratio)
        new_edge_index[2, mask] *= -1
        return torch.from_numpy(new_edge_index).long()
    elif augment == 'delete':
        mask = generate_1d_mask(perspective.shape[1], args.mask_ratio)
        return torch.from_numpy(new_edge_index[:, ~mask]).long()
    elif augment == 'add':
        # number of edges to add
        to_add = int(new_edge_index.shape[1] * args.mask_ratio)
        # get edge index and labels
        edge_index = torch.from_numpy(new_edge_index[0:2, :])
        labels = torch.from_numpy(new_edge_index[2, :])
        # create fake edges and labels
        add_edges = negative_sampling(edge_index, num_neg_samples=to_add)  # create fake edges
        add_labels = torch.from_numpy(np.where(np.random.uniform(0, 1, to_add) >= .5, 1, -1))  # create fake labels
        edge_index = torch.cat([edge_index, add_edges], dim=1)
        labels = torch.cat([labels, add_labels], dim=0)
        return torch.cat([edge_index, labels.view(1, len(labels))], dim=0).long()
    else:
        raise Exception('Unsupported method.')
