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



def embed_text(text, model_name='roberta-base', output_dim=128):
    # Load the RoBERTa model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    # Define a projection layer to change the output tensor dimension
    projection_layer = nn.Linear(model.config.hidden_size, output_dim)

    # Prepare the input
    inputs = tokenizer(text, return_tensors='pt')

    # Perform forward pass
    outputs = model(**inputs)

    # Use the projection layer to change the output tensor dimension
    reshaped_outputs = projection_layer(outputs.last_hidden_state[:, 0, :])

    # Return the output tensor
    return reshaped_outputs



if __name__ == "__main__":
    # load data
    dataset = load_dataset(opt.data_dir + opt.subject + "/Questions_CourseX.xlsx")
    # print(dataset)
    for row in dataset:
        row_str = ' '.join(map(str, row))
        encoder = embed_text(row_str)
        print(encoder)



