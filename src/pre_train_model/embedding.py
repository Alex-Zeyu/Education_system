import time
import numpy as np
import pandas as pd
import re
from config import opt
from bs4 import BeautifulSoup


import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

# Document Embeddings
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, DocumentRNNEmbeddings, TransformerDocumentEmbeddings
# https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md
from flair.data import Sentence

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

# load data
def load_dataset(data_frame, data_path):
    cols = opt.semantic_features + opt.handcrafted_features
    if data_frame is None:
        data_frame = pd.read_excel(data_path, usecols=cols)
    else:
        data_frame = data_frame[cols]
    dataset = data_frame.values

    # normalize opt.handcrafted_features
    if opt.model_type != "None":
        handcrafted_features = dataset[:, 1 + len(opt.semantic_features):]
        for idx in range(np.size(handcrafted_features, 1)):
            handcrafted_features[:, idx] = handcrafted_features[:, idx] / handcrafted_features[:, idx].max()
        dataset[:, 1+len(opt.semantic_features):] = handcrafted_features

    return dataset

# the average of all word embeddings
def get_DocumentPoolmbeddings(input_text):
    glove_embedding = WordEmbeddings('glove')
    document_embeddings = DocumentPoolEmbeddings([glove_embedding])
    sentence = Sentence(input_text)
    document_embeddings.embed(sentence)
    features = sentence.get_embedding()

    return features

# run an RNN over all words in sentence and use the final state of the RNN as embedding for the whole document
def get_DocumentRNNEmbeddings(input_text):
    glove_embedding = WordEmbeddings('glove')
    document_embeddings = DocumentRNNEmbeddings([glove_embedding])
    sentence = Sentence(input_text)
    # embed the sentence with document embedding
    document_embeddings.embed(sentence)
    features = sentence.get_embedding()

    return features

def get_TransformerDocumentEmbeddings(input_text):
    # init embedding
    embedding = TransformerDocumentEmbeddings('roberta-base')
    # create a sentence
    sentence = Sentence(input_text)
    # embed the sentence
    embedding.embed(sentence)
    features = sentence.get_embedding()
    return features

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

def clean_html(raw_html):
    if not isinstance(raw_html, str):
        return "", ""
    # print("raw_html: " + raw_html)

    # 1. get all special html tags
    soup = BeautifulSoup(raw_html, "html.parser")
    tag_list = [tag.name for tag in soup.find_all()]
    tag_list = set(tag_list)
    tag_list.discard("p")
    str_tags = str(tag_list) + " " if len(tag_list) > 0 else ""

    # 2. replace special html string with space
    # 2.1 replace "&nbsp;" with one space
    raw_html = re.sub('&nbsp;', ' ', raw_html)
    # 2.2 common html tags: mainly for replacing <br/> with space
    raw_html = re.sub(r'<.*?>', ' ', raw_html)

    # 3. remove other html tags with beautiful soup 4
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text()

    # 4. uniform user input
    # 4.1 multiple . or _ (usually in stem to represent answer)
    text = re.sub(r'(\.)\1+', r' ____ ', text)
    text = re.sub(r'(_)\1+', r' ____ ', text)
    # 4.2. multiple -
    text = re.sub(r'(-)\1+', r' -- ', text)

    # 5. clear sentence
    # 5.1 multiple whitespace
    text = re.sub(' +', ' ', text).strip()
    # # 5.2 sentence end
    text = text if re.search(r'[.?!]$', text) else text + "."

    # print("text: " + text)
    return text


if __name__ == "__main__":
    # load data
    dataset = load_dataset(opt.data_dir + opt.subject + "/Questions_CourseX.xlsx")
    # print(dataset)
    for row in dataset:
        row_str = ' '.join(map(str, row))
        row_str = clean_html(row_str)
        if 'glove_meanpool' in opt.emb_method:
            encoder = get_DocumentPoolmbeddings(row_str) #output 100 dim
        elif opt.emb_method == 'glove_rnn':
            encoder = get_DocumentRNNEmbeddings(row_str)
        elif opt.emb_method == 'roberta-base_raw':
            encoder = get_roberta_embeddings(row_str)
        elif opt.emb_method == 'roberta-base_document':
            encoder = get_TransformerDocumentEmbeddings(row_str) #output 768 dim
        print(encoder)



