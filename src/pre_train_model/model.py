import torch
import torch.nn as nn
import numpy as np

from encoder import Encoder
import flair
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, TransformerDocumentEmbeddings
# https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md
from torchnlp.nn import Attention



class Model(nn.Module):
    def __init__(self, opt):

        super(Model, self).__init__()
        self.opt = opt
        flair.device = torch.device(self.opt.device)

        if 'glove' in self.opt.emb_method:
            self.init_glove()
        elif self.opt.emb_method == 'sbert':
            self.init_sbert()
        elif self.opt.emb_method == 'roberta':
            self.init_roberta()

        # init semantic_out_size
        semantic_out_size = opt.out_size
        if self.opt.enable_attention_scorer and "Solo" not in self.opt.model_type:
            self.attention = Attention(self.word_dim, attention_type='general')
            self.weight_encoder = Encoder(opt.semantic_enc_method,
                                          len(self.opt.semantic_features),
                                          self.opt.hidden_size,
                                          self.opt.out_size)
            semantic_out_size = opt.out_size * 2

        self.encoder = Encoder(opt.semantic_enc_method, self.word_dim, opt.hidden_size, opt.out_size)

        if "Encoded" in self.opt.model_type:
            # handcrafted feature encode
            self.handcrafted_features_encoder = Encoder(opt.handcrafted_features_encode_method,
                                                        1,
                                                        opt.hidden_size,
                                                        opt.out_size)
            self.predict = nn.Linear(opt.out_size if "Solo" in self.opt.model_type
                                     else (opt.out_size + semantic_out_size),
                                     opt.num_labels)
        elif "Raw" in self.opt.model_type:
            # handcrafted feature concat directly
            self.predict = nn.Linear(len(self.opt.handcrafted_features) if "Solo" in self.opt.model_type
                                     else (semantic_out_size + len(self.opt.handcrafted_features)), opt.num_labels)
        elif "EDF" == self.opt.model_type:
            # SCQC + handcrafted feature
            self.predict = nn.Linear(opt.out_size + len(self.opt.handcrafted_features), opt.num_labels)
        else:
            # no handcrafted feature
            self.predict = nn.Linear(semantic_out_size, opt.num_labels)

        nn.init.uniform_(self.predict.weight, -0.1, 0.1)
        nn.init.uniform_(self.predict.bias, -0.1, 0.1)
        self.dropout = nn.Dropout(self.opt.dropout)

    def forward(self, x):
        # 1. get semantic embedding of each MCQ component, get handcrafted features
        components_embs, features = self.get_embedding(x)
        if hasattr(self, 'attention') and components_embs is not None:
            # attention layer
            _, weights = self.attention(components_embs, components_embs)

        # 2. semantic features
        if "Solo" not in self.opt.model_type:
            # encode multiple MCQ components semantic embeddings to one MCQ embedding
            x = self.encoder(components_embs.to(self.opt.device))
            x = self.dropout(x)
            if hasattr(self, 'weight_encoder'):
                components_coefficient_embds = self.weight_encoder(weights.to(self.opt.device))
                x = components_coefficient_embds if "EDF" == self.opt.model_type \
                    else torch.cat((x, components_coefficient_embds), 1)

        # 3. handcrafted features
        features = features.astype(float)
        if "Encoded" in self.opt.model_type:
            features = torch.FloatTensor(features).unsqueeze(2)
            if self.opt.use_gpu:
                features = features.to(self.opt.device)
            features_embds = self.handcrafted_features_encoder(features.to(self.opt.device))
        else:
            features_embds = torch.FloatTensor(features).to(self.opt.device)

        # 4. concat MCQ semantic features with handcrafted features
        if self.opt.model_type == "None":
            x = x
        elif "Solo" in self.opt.model_type:
            x = features_embds
        else:
            x = torch.cat((x, features_embds), 1)

        # predict rating
        x = self.predict(x)
        return x

    def init_glove(self):
        '''
        load the GloVe model
        '''
        if self.opt.emb_method == 'glove':
            # init flair embedding
            glove_embedding = WordEmbeddings('glove')
            # initialize the document embeddings, fine_tune_mode = linear, pooling = mean works the best
            # default: fine_tune_mode="none", pooling='mean'
            self.embedding_model = DocumentPoolEmbeddings([glove_embedding])

        self.word_dim = self.opt.glove_dim

    def init_sbert(self):
        '''
        initilize the Flair SBert model
        '''
        self.embedding_model = TransformerDocumentEmbeddings(self.opt.sbert_pretrained_model)
        self.word_dim = self.opt.sbert_dim


    def init_roberta(self):
        '''
        initilize the RoBERTa model
        '''
        self.embedding_model = TransformerDocumentEmbeddings(self.opt.roberta_pretrained_model)
        # the test result shows that:
        #   when only use the first layer of RoBERTa, the performance is much better than use all 13 layers
        # self.embedding_model = TransformerDocumentEmbeddings(self.opt.roberta_pretrained_model, layers="0")
        self.word_dim = self.opt.roberta_dim


    def get_embedding(self, question_batch):
        '''
        get the mean glove embedding vectors for sentences
        '''
        question_batch = np.asarray(question_batch)
        # split sentences_lists into MCQ components and 1 additional feature list:
        components_length = len(self.opt.semantic_features)

        components_batch = question_batch[:, :components_length]
        #features_batch = question_batch[:, components_length:]

        batch_components_embd = []
        # if "Solo" in self.opt.model_type:
        #     return None, features_batch

        # get semantic embedding for components of a MCQ
        for components in components_batch:
            components_embd = []
            for component in components:
                sentence = Sentence(component)
                self.embedding_model.embed(sentence)
                components_embd.append(sentence.embedding)
            components_embd = torch.stack(components_embd)
            batch_components_embd.append(components_embd)

        batch_components_embd = torch.stack(batch_components_embd)

        if hasattr(self, 'finetuned_glove_encoder'):
            batch_components_embd = self.finetuned_glove_encoder(batch_components_embd)
            batch_components_embd = batch_components_embd.contiguous()

        return batch_components_embd
