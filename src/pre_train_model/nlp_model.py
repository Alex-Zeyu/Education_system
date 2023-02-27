import torch
from transformers import RobertaTokenizer, RobertaModel


class NLPModel(torch.nn.Module):
    def __init__(self, max_length=512, feature_size=128, device='cuda'):
        super(NLPModel, self).__init__()
        self.max_length = max_length
        self.feature_size = feature_size
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')

    def forward(self, x: str):
        encoded_input = self.tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        ).to(self.device)

        # with torch.no_grad():
        model_output = self.model(**encoded_input)

        features = model_output.last_hidden_state[:, 0, :]
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        features = features.squeeze(0)
        features = features[:self.feature_size]

        return features
