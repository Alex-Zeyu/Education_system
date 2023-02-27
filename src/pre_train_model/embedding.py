import time
import numpy as np
import pandas as pd
from config import opt


import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

# load data
def load_dataset(data_path):
    cols = opt.semantic_features + opt.handcrafted_features
    data_frame = pd.read_excel(data_path, usecols=cols)
    dataset = data_frame.values

    # normalize opt.handcrafted_features
    if opt.model_type != "None":
        handcrafted_features = dataset[:, 1 + len(opt.semantic_features):]
        for idx in range(np.size(handcrafted_features, 1)):
            handcrafted_features[:, idx] = handcrafted_features[:, idx] / handcrafted_features[:, idx].max()
        dataset[:, 1+len(opt.semantic_features):] = handcrafted_features

    return dataset



def get_roberta_embeddings(input_text, max_length=512, feature_size=128):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    encoded_input = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    features = model_output.last_hidden_state[:, 0, :]
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    features = features.squeeze(0)
    features = features[:feature_size]

    return features



if __name__ == "__main__":
    # load data
    dataset = load_dataset(opt.data_dir + opt.subject + "/Questions_CourseX.xlsx")
    # print(dataset)
    for row in dataset:
        row_str = ' '.join(map(str, row))
        encoder = get_roberta_embeddings(row_str)
        print(encoder)



