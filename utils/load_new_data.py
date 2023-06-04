import numpy as np
import pandas as pd

"""
The new datasets are: Sydney, Sydney_AdditionalLTISet and Cardiff, the Sydney_AdditionalLTISet dataset is too small 
(3532 edges for the largest course) thus is ignored.
"""


def load_csv(path: str) -> pd.DataFrame:
    def get_title() -> list[str]:
        with open(path) as f:
            f.readline()
            t = f.readline().split("|")
            return list(filter(None, list(map(str.strip, t))))  # strip the title and remove empty element

    title = get_title()
    n_cols = len(title)

    df = pd.read_csv(path, sep="|", usecols=range(1, n_cols + 1), header=0, names=title, dtype=object,
                     encoding="unicode-escape")
    df.drop([0, 1, df.shape[0] - 1], axis=0, inplace=True)  # drop the line seperator (e.g. +--------+)
    df.reset_index(drop=True, inplace=True)
    return df


def load_answer(path: str):
    type_dict = {
        'user': np.int64,
        'question_id': np.int64,
        'answer': str,
        'course_id': np.int64
    }
    answer_df = load_csv(path)[['user', 'question_id', 'answer', 'course_id']].astype(type_dict)
    answer_df['answer'] = answer_df['answer'].str.strip()
    return answer_df


def load_question(path: str, responses: int):
    type_dict = {
        'id': np.int64,
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
    question_df = load_csv(path)
    question_df = question_df[~question_df['id'].isna()]  # filter the NA rows
    question_df = question_df.drop(
        columns=['timestamp', 'user', 'top_rating_count', 'total_comments', 'course_id']).astype(type_dict)
    # ignore any hidden questions
    question_df = question_df[(question_df['deleted'] == 0) & (question_df['total_responses'] >= responses)].drop(
        columns=['deleted']).reset_index(drop=True)
    question_df['answer'] = question_df['answer'].str.strip()
    question_df.rename(columns={'id': 'question_id'}, inplace=True)

    return question_df


def get_merged(answer_df, question_df, course_id=None, suffix=''):
    merged_df = answer_df.merge(question_df, on=['question_id'])
    merged_df['sign'] = np.where(merged_df['answer_x'] == merged_df['answer_y'], 1, -1)
    merged_df.drop(columns=['answer_x', 'answer_y'], inplace=True)

    # subset data according to course_id
    if course_id:
        merged_df = merged_df[merged_df['course_id'] == course_id]
        merged_df.reset_index(drop=True, inplace=True)

    info = {
        'user_num': merged_df['user'].nunique(),
        'ques_num': merged_df['question_id'].nunique(),
        'edge_num': merged_df.shape[0],
        'pos_edge': (merged_df['sign'] > 0).sum(),
        'neg_edge': (merged_df['sign'] < 0).sum()
    }

    # save dictionary
    import pickle
    save_path = os.path.join('..', 'datasets', 'processed', f'{args.dataset}{suffix}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'data_info.pkl'), 'wb') as f:
        pickle.dump(info, f)

    # reindex user and question id
    from sklearn.preprocessing import LabelEncoder
    user_encoder = LabelEncoder()
    ques_encoder = LabelEncoder()
    merged_df['user'] = user_encoder.fit_transform(merged_df['user'])
    merged_df['question_id'] = ques_encoder.fit_transform(merged_df['question_id']) + info['user_num']

    return merged_df, info


def save_edge_index(df: pd.DataFrame, args, suffix=''):
    import os
    import torch
    import random

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # save edge index as local files
    edge_index = torch.tensor(df[['user', 'question_id', 'sign']].values)
    path = os.path.join('..', 'datasets', 'processed', f'{args.dataset}{suffix}')  # folder path
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


def load_nlp_emb(path: str, df: pd.DataFrame = None, method: str = None, suffix: str = ''):
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
        save_path = os.path.join('..', 'embedding', f'{args.dataset}{suffix}')  # folder path
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
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--splits', type=int, default=10, help='How many times to split the dataset.')
    parser.add_argument('--dataset', type=str, default='Sydney', help='The dataset to be used.')
    parser.add_argument('--responses', type=int, default=1,
                        help='Only keep the questions with responses >= certain number')
    args = parser.parse_args()
    print(args)

    # file path
    answer_path = os.path.join('..', 'datasets', 'Sydney_Cardiff_PW_Data', args.dataset, 'All_Answers.txt')
    question_path = os.path.join('..', 'datasets', 'Sydney_Cardiff_PW_Data', args.dataset, 'All_Questions.txt')

    answer, question = load_answer(answer_path), load_question(question_path, args.responses)
