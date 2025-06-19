import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# defining utility functions to get BERT tokenizer and model
def get_tokenizer(model_name='bert-base-uncased'):
    return BertTokenizer.from_pretrained(model_name)

def get_model(model_name='bert-base-uncased', num_labels=2):
    return BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

class SentimentBERT(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super(SentimentBERT, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

