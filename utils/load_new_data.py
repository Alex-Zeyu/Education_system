import numpy as np
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    def get_title(path: str) -> list[str]:
        with open(path) as f:
            f.readline()
            title = f.readline().split("|")
            title = list(filter(None, list(map(str.strip, title))))  # strip the title and remove empty element
            return title

    title = get_title(path)
    n_cols = len(title)

    df = pd.read_csv(path, sep="|", usecols=range(1, n_cols + 1), header=0, names=title, dtype=object,
                     encoding="unicode-escape")
    # drop the line seperator (e.g. +--------+)
    df.drop([0, 1, df.shape[0] - 1], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_answer(path: str):
    type_dict = {
        'user': np.int64,
        'question_id': np.int64,
        'answer': str,
        'course_id': np.int64
    }
    answer = load_csv(path)[['user', 'question_id', 'answer', 'course_id']].astype(type_dict)
    answer['answer'] = answer['answer'].str.strip()
    return answer


def load_question(path: str, responses: int):
    type_dict = {
        'id': np.int64,
        'course_id': np.int64,
        'avg_rating': np.float64,
        'total_responses': np.int64,
        'total_ratings': np.int64,
        'avg_difficulty': np.float64,
        'deleted': np.int64,
        'answer': str,
        'numAlts': np.int64,
        'question': str,
        'altA': str,
        'altB': str,
        'altC': str,
        'altD': str,
        'altE': str
    }
    question = load_csv(path).drop(columns=['timestamp', 'user', 'top_rating_count', 'total_comments']).astype(
        type_dict)
    # ignore any questions that are edited
    question = question[(question['deleted'] == 0) & (question['total_responses'] >= responses)].drop(
        columns=['deleted'])
    question.reset_index(drop=True, inplace=True)
    question['answer'] = question['answer'].str.strip()
    question.rename(columns={'id': 'question_id'}, inplace=True)

    return question


def get_merged(answer_df, question_df):
    merged = answer_df.merge(question_df, on=['question_id'])
    merged['sign'] = np.where(merged['answer_x'] == merged['answer_y'], 1, -1)
    merged.drop(columns=['answer_x', 'answer_y', 'course_id_x', 'course_id_y'], inplace=True)

    data_info = {
        'user_num': merged['user'].nunique(),
        'ques_num': merged['question_id'].nunique(),
        'edge_num': merged.shape[0],
        'pos_edge': (merged['sign'] > 0).sum(),
        'neg_edge': (merged['sign'] < 0).sum()
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
    merged['user'] = user_encoder.fit_transform(merged['user'])
    merged['question_id'] = ques_encoder.fit_transform(merged['question_id']) + data_info['user_num']

    return merged, data_info


def save_edge_index(df: pd.DataFrame, args):
    import os
    import torch
    import random

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # generate mask for splitting
    def generate_mask():
        mask = torch.ones(df.shape[0], dtype=torch.bool)
        mask[torch.randperm(mask.size(0))[:int(args.test_ratio * mask.size(0))]] = 0
        return mask

    # save edge index as local files
    edge_index = torch.tensor(df[['user', 'question_id', 'sign']].values)
    path = os.path.join('..', 'datasets', 'processed', args.dataset)
    if not os.path.exists(path):
        os.makedirs(path)

    for split_i in range(args.splits):
        mask = generate_mask()

        train_ei = edge_index[mask]
        test_ei = edge_index[~mask]

        # undirected graph
        train_ei = torch.cat([train_ei[:, [0, 1, 2]], train_ei[:, [1, 0, 2]]], dim=0).T
        test_ei = torch.cat([test_ei[:, [0, 1, 2]], test_ei[:, [1, 0, 2]]], dim=0).T

        torch.save(train_ei, os.path.join(path, f'train_{split_i}.pt'))
        torch.save(test_ei, os.path.join(path, f'test_{split_i}.pt'))


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

        nlp_cols = ['question', 'altA', 'altB', 'altC', 'altD', 'altE', 'explanation']
        qus_valid = df.drop_duplicates(subset=['question_id']).sort_values(by=['question_id'], ignore_index=True)
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


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--splits', type=int, default=2, help='How many times to split the dataset.')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Split the training and test set.')
    parser.add_argument('--dataset', type=str, default='Sydney_AdditionalLTISet', help='The dataset to be used.')
    parser.add_argument('--responses', type=int, default=20,
                        help='Only keep the questions with responses >= certain number')
    args = parser.parse_args()
    print(args)

    # file path
    answer_path = os.path.join('..', 'datasets', 'Sydney_Cardiff_PW_Data', args.dataset, 'All_Answers.txt')
    question_path = os.path.join('..', 'datasets', 'Sydney_Cardiff_PW_Data', args.dataset, 'All_Questions.txt')

    answer, question = load_answer(answer_path), load_question(question_path, args.responses)
    merged, data_info = get_merged(answer, question)
    # _ = load_nlp_emb("none", merged, "glove")
