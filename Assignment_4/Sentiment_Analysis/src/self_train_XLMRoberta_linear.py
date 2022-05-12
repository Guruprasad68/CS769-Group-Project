# pip install transformers
# pip install torch==1.5.0
# pip install torchtext==0.6.0

from torch.nn.modules import dropout
import torch
import sys
import random
import numpy as np
import time
from transformers import AutoTokenizer, AutoModel
from torchtext import datasets
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score
import torch

import torch.optim as optim
from sklearn.metrics import confusion_matrix

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




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#print(f'The {model} has {count_parameters(model):,} trainable parameters')
#print("Parameters for " + f'{model}')

# for name, param in model.named_parameters():                
#     if param.requires_grad:
#         #print(name)
#         pass



def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)




def categorical_accuracy(preds, y):
    count0,count1,count2 = torch.zeros(1),torch.zeros(1),torch.zeros(1)
    total0,total1,total2 = torch.FloatTensor(1),torch.FloatTensor(1),torch.FloatTensor(1)
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    predictions = max_preds.squeeze(1)
    true_correct = [0,0,0]
    for j,i in enumerate(y.cpu().numpy()):
      true_correct[y.cpu().numpy()[j]]+=1
      if i==0:
        count0+=correct[j]
        total0+=1
      elif i==1:
        count1+=correct[j]
        total1+=1
      elif i==2:
        count2+=correct[j]
      else:
        total2+=1
    metric=torch.FloatTensor([count0/true_correct[0],count1/true_correct[1],count2/true_correct[2],
                              f1_score(y.cpu().detach().numpy(),
                                       predictions.cpu().detach().numpy(),average='macro')])
    return correct.sum() / torch.FloatTensor([y.shape[0]]),metric,confusion_matrix(y.cpu().detach().numpy(),max_preds.cpu().detach().numpy())



def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        
        predictions =  model(batch.text).squeeze(1)
         
        loss = criterion(predictions, batch.label)

        acc,_,_ = categorical_accuracy(predictions, batch.label)
        
        loss.backward()
        clip_gradient(model, 1e-1)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_all_acc = torch.FloatTensor([0,0,0,0])
    confusion_mat = torch.zeros((3,3))
    confusion_mat_temp = torch.zeros((3,3))

    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc,all_acc,confusion_mat_temp = categorical_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_all_acc += all_acc
            confusion_mat+=confusion_mat_temp

    return epoch_loss / len(iterator), epoch_acc / len(iterator),epoch_all_acc/len(iterator),confusion_mat


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens


def main():

    # Hyperparameters
  BATCH_SIZE = 64
  lr = 1e-6
  dropout = 0.3
  hidden_size = 50
  n_classes = 3
  N_EPOCHS = 20

  # BERT model title 
  bert_model_title = "xlm-roberta-base"

  data_path = "data/self_train_data"
  train_name = "hinglish_concat_train.txt"
  valid_name = "hinglish_dev_unt_text.txt"
  test_name = "hinglish_test_unt_text.txt"

  model_save_name = "../checkpoint/self_train_xlmRoberta.txt"

  SEED = 1234
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("Device in use:",device)

  tokenizer = AutoTokenizer.from_pretrained(bert_model_title)
  print(f'{bert_model_title} Tokenizer Loaded...')

  init_token_idx = tokenizer.cls_token_id
  eos_token_idx = tokenizer.sep_token_id
  pad_token_idx = tokenizer.pad_token_id
  unk_token_idx = tokenizer.unk_token_id

  max_input_length = 150
  print("Max input length: %d" %(max_input_length))

  UID = data.Field(sequential=False, use_vocab=False, pad_token=None)
  TEXT = data.Field(batch_first = True,
                    use_vocab = False,
                    tokenize = tokenize_and_cut,
                    preprocessing = tokenizer.convert_tokens_to_ids,
                    init_token = init_token_idx,
                    eos_token = eos_token_idx,
                    pad_token = pad_token_idx,
                    unk_token = unk_token_idx)


  LABEL = data.LabelField()

  fields = [('uid',UID),('text', TEXT),('label', LABEL)]
  train_data, valid_data, test_data = data.TabularDataset.splits(
                                          path = data_path,
                                          train = train_name,
                                          validation = valid_name,
                                          test = test_name,
                                          format = 'tsv',
                                          fields = fields,
                                          skip_header = True)

  print('Data loading complete')
  print(f"Number of training examples: {len(train_data)}")
  print(f"Number of validation examples: {len(valid_data)}")
  print(f"Number of test examples: {len(test_data)}")

  # tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[0])['text'])


  LABEL.build_vocab(train_data, valid_data)
  print(LABEL.vocab.stoi)

  train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
      (train_data, valid_data, test_data),
      sort_key=lambda x: len(x.text), 
      batch_size = BATCH_SIZE, 
      device = device)

  print('Iterators created')

  print(f'Downloading {bert_model_title} model...')

  bert = AutoModel.from_pretrained(bert_model_title)

  print(f'{bert_model_title} model downloaded')

  model = BERTLinearSentiment(bert, 
                              dropout, 
                              hidden_size, 
                              n_classes, 
                              freeze_bert=False)

  # Load from Checkpoint
  path = "../checkpoint/self_train_xlmRoberta.txt"

  if torch.cuda.is_available():
    model.load_state_dict(torch.load(path))
  else:
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))


  optimizer = optim.Adam(model.parameters(),lr)

  criterion = nn.CrossEntropyLoss()
  nll_loss = nn.NLLLoss()
  log_softmax = nn.LogSoftmax()

  model = model.to(device)
  criterion = criterion.to(device)
  nll_loss = nll_loss.to(device)
  log_softmax = log_softmax.to(device)

  # Train the model

  best_f1 = -1

  for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc,tot,conf = evaluate(model, valid_iterator, criterion)
    f1 = tot[3]
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if f1 > best_f1:
        best_f1 = f1
        
        path = model_save_name
        print(path)
        torch.save(model.state_dict(), path)
    
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    print(tot)
    print(conf)

  # Load the Best Model and evaluate

  path = "../checkpoint/self_train_xlmRoberta.txt"

  if torch.cuda.is_available():
    model.load_state_dict(torch.load(path))
  else:
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

  valid_loss, valid_acc, tot, cmat = evaluate(model, test_iterator, criterion)

  print(f'\t Test. Loss: {valid_loss:.3f} |  Test. Acc: {valid_acc*100:.2f}%')
  print(tot)
  print("Confusion Matrix:\n",cmat)


if __name__ == "__main__":
  main()