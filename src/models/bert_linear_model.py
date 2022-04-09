import torch
import torch.nn as nn
import torch.nn.functional as F


# BERT + Linear CLassifier for Sentiment Analysis

class BERTLinearSentiment(nn.Module):
    def __init__(self,
                 bert,
                 dropout,
                 hidden_size,
                 n_classes,
                 freeze_bert = False
                 ):
        
        super().__init__()
        
        # Instantiate BERT model
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.n_classes)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input):
        
        #input = [batch size, sent len]
        
        # Feed input to BERT
        outputs = self.bert(input_ids=input)  

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits