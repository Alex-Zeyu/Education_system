import numpy as np
import pandas as pd


def load_csv(ans_path: str, qus_path: str):
    answer = pd.read_excel(ans_path)
    question = pd.read_excel(qus_path)
    question = question[question['TotalResponses'] > 0]

    merged = answer[['UserID', 'QuestionID', 'Answer']].merge(question.drop(columns=['UserID']), on='QuestionID')
    merged['Sign'] = np.where(merged['Answer_x'] == merged['Answer_y'], 1, -1)
    merged.drop(columns=['Answer_x', 'Answer_y'], inplace=True)

    # user and question numbers
    data_info = {
        'user_num': merged['UserID'].nunique(),
        'ques_num': merged['QuestionID'].nunique(),
        'edge_num': merged.shape[0],
        'pos_edge': (merged['Sign'] > 0).sum(),
        'neg_edge': (merged['Sign'] < 0).sum()
    }

    # save dictionary
    import pickle
    save_path = os.path.join('..', 'datasets', 'processed', args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'data_info.pkl'), 'wb') as f:
        pickle.dump(data_info, f)

    # reindex user and question id
    from sklearn.preprocessing import LabelEncoder
    user_encoder = LabelEncoder()
    ques_encoder = LabelEncoder()
    merged['UserID'] = user_encoder.fit_transform(merged['UserID'])
    merged['QuestionID'] = ques_encoder.fit_transform(merged['QuestionID']) + data_info['user_num']

    return answer, question, merged, data_info


def save_edge_index(df: pd.DataFrame, args):
    import os
    import torch
    import random

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # save edge index as local files
    edge_index = torch.tensor(df[['UserID', 'QuestionID', 'Sign']].values)
    path = os.path.join('..', 'datasets', 'processed', args.dataset)
    if not os.path.exists(path):
        os.makedirs(path)

    for split_i in range(args.splits):
        # generate mask
        train_idx, val_idx, test_idx = torch.utils.data.random_split(range(edge_index.size(0)), [.85, .05, .1])
        train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

        train_edge_index, val_edge_index, test_edge_index = edge_index[train_idx], edge_index[val_idx], edge_index[
            test_idx]

        # directed graph
        torch.save(train_edge_index.t(), os.path.join(path, f'train_{split_i}.pt'))
        torch.save(val_edge_index.t(), os.path.join(path, f'val_{split_i}.pt'))
        torch.save(test_edge_index.t(), os.path.join(path, f'test_{split_i}.pt'))


def load_edge_index(dataset: str, train: bool, round: int):
    """invoke in the root directory"""
    import os
    import torch
    prefix = "train_" if train else "test_"
    data_path = os.path.join('datasets', 'processed', dataset, f'{prefix}{round}.pt')
    return torch.load(data_path)


def load_nlp_emb(path: str, df: pd.DataFrame = None, method: str = None):
    import os
    import torch

    if os.path.exists(path):
        nlp_emb = torch.load(path)
    else:
        import re
        from bs4 import BeautifulSoup
        import unicodedata
        from flair.data import Sentence

        nlp_cols = ['Question', 'OptionA', 'OptionB', 'OptionC', 'OptionD', 'OptionE', 'Explanation', 'Tags']
        qus_valid = df.drop_duplicates(subset=['QuestionID']).sort_values(by=['QuestionID'], ignore_index=True)
        qus_valid = qus_valid[nlp_cols].values.astype(str)

        # path to save embeddings
        save_path = os.path.join('..', 'embedding', args.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # clean html
        def clean_html(x):
            if not isinstance(x, str):
                return ''
            x = BeautifulSoup(x, "html.parser").text
            x = unicodedata.normalize('NFKC', x).strip()
            x = re.sub(r'_+', ' ', x)
            x = re.sub(r'-+', ' ', x)
            x = re.sub(r'\s{2,}', ' ', x)
            return x

        if method == "roberta":
            from flair.embeddings import TransformerDocumentEmbeddings
            nlp_encoder = TransformerDocumentEmbeddings('roberta-base')
            save_path = os.path.join(save_path, 'roberta.pt')

        elif method == "glove":
            from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
            nlp_encoder = DocumentPoolEmbeddings([WordEmbeddings('glove')], fine_tune_mode='none')
            save_path = os.path.join(save_path, 'glove.pt')

        else:
            raise Exception("Unsupported embedding method.")

        qus_sent_arr = [Sentence(clean_html(''.join(row))) for row in qus_valid]

        for s in qus_sent_arr:
            nlp_encoder.embed(s)
        nlp_emb = torch.cat([s.embedding.unsqueeze(0) for s in qus_sent_arr], dim=0)
        torch.save(nlp_emb, save_path)

    return nlp_emb


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--splits', type=int, default=10, help='How many times to split the dataset.')
    parser.add_argument('--dataset', type=str, default='Biology', help='The dataset to be used.')
    args = parser.parse_args()
    print(args)

    # file path
    answer_path = os.path.join('..', 'datasets', 'PeerWiseData', args.dataset, 'Answers_CourseX.xlsx')
    question_path = os.path.join('..', 'datasets', 'PeerWiseData', args.dataset, 'Questions_CourseX.xlsx')

    answer, question, merged, data_info = load_csv(answer_path, question_path)
    # save_edge_index(merged, args)
    # load_nlp_emb("none", merged, "glove")
