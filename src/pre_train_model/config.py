import torch

class Config:
    # 0. gpu
    use_gpu = False
    gpu_id = 0
    device = torch.device("cuda:{}".format(gpu_id) if (torch.cuda.is_available() and use_gpu) else "cpu")

    # 1. model type
    # EDF-GloVe-SCQC: model_type="EDF" + emb_method="glove" + enable_attention_scorer="True"
    # EDF-QDQE-SCQC: model_type="EDF" + emb_method="glove_finetuned" + enable_attention_scorer="True"
    model_type = "None"  # "None" / "Encoded" / "Raw" / "Solo-Raw" / "Solo-Encoded" / "EDF"

    # 2. dataset
    data_dir = "../../datasets/PeerWiseData/"
    subject = "Biology"  # Biology, Law

    # 3. semantic features
    # ['AvgQuality', 'TotalResponses', 'TotalRatings', 'Answer', 'Num_options', 'Question', 'OptionA', 'OptionB', 'OptionC', 'OptionD', 'OptionE', 'Explanation', 'Tags']
    semantic_features = ['AvgQuality', 'TotalResponses', 'TotalRatings', 'Answer', 'Num_options', 'Question', 'OptionA', 'OptionB', 'OptionC', 'OptionD', 'OptionE', 'Explanation', 'Tags']

    emb_method = 'glove'  # glove/roberta-base_raw/roberta-base_Document
    semantic_enc_method = 'transformer'  # rnn/transformer
    enable_attention_scorer = False  # True to enable transformer layer to calculate attention score and encode the score as additional features

    # 4. handcrafted features
    numWords = ['NumWords_Question', 'NumWords_Explanation', 'NumWords_OptionA', 'NumWords_OptionB', 'NumWords_OptionC', 'NumWords_OptionD', 'NumWords_OptionE']
    readability = ['Flesch_readability', 'FleschKincaid_readability', 'GunningFog_readability', 'ColemanLiau_readability', 'LinsearWrite_readability', 'AutomatedReadabilityIndex_readability', 'Spache_readability', 'DaleChall_readability', 'Smog_readability']
    # distribution = ['OptionA_percentage', 'OptionB_percentage', 'OptionC_percentage', 'OptionD_percentage', 'OptionE_percentage']

    # please don't change the order when print_weight = True
    handcrafted_features = ['Num_options'] + numWords + ['Grammar'] + readability

    if model_type == "None":
        # no handcrafted_features
        handcrafted_features = []
        handcrafted_features_encode_method = None
    elif model_type == "Encoded" or model_type == "Solo-Encoded":
        # transformer as encoder layer of handcrafted_features
        handcrafted_features_encode_method = "transformer"
    elif "Raw" in model_type or model_type == "EDF":
        # no encoder layer of handcrafted_features
        handcrafted_features_encode_method = None
    else:
        handcrafted_features = []
        handcrafted_features_encode_method = None

    # 5. embedding parameters
    # glove
    glove_dim = 100
    glove_pretrained_model = 'glove'
    glove_enc_method = 'transformer'  # rnn/transformer
    glove_finetuned_model = './QDQE/result/saved_models/QDQE-finetuned/' + subject + '/' + glove_enc_method\
                            + '/best_epoch/model.pt'

    # SBert
    sbert_dim = 768
    sbert_pretrained_model = 'roberta-base'
    sbert_finetuned_model = './QDQE/result/saved_models/sbert_roberta_finetuned/'

    # RoBERTa
    roberta_dim = 768
    roberta_pretrained_model = 'roberta-base'
    roberta_finetuned_model = './QDQE/result/saved_models/RoBERTa-bio/epoch_1/'

    # 6. encode parameters
    # TODO 3 these 2 parameters could be different for semantic and handcrafted features
    hidden_size = 200
    out_size = 64

    # 7. train parameters
    batch_size = 16
    epochs = 50
    predict_correctness_threshold = 0.25

    num_labels = 1
    seed = 2021
    dropout = 0.5
    lr = 1e-3
    weight_decay = 1e-4
    print_weight = False


def parse(self, kwargs):
    '''
    user can update the default hyperparamter
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            raise Exception('opt has No key: {}'.format(k))
        setattr(self, k, v)

    print('*************************************************')
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print("{} => {}".format(k, getattr(self, k)))

    print('*************************************************')


Config.parse = parse
opt = Config()
