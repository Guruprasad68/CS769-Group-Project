## Assignment 4: Self-Training for Sentiment Analysis and Named-Entity-Recognition of Hindi-English Code Switched Data
In this asssignment we study two tasks for Hindi-English Code-Switched Data:
1. Sentiment Analysis
2. Named Entity Recognition

### Sentiment Analysis
For Sentiment Analysis, we finetune the pre-trained models of XLM-RoBERTa-Base, mBERT and RoBERTa-Base on the train data of the "[SemEval-2020 Task 9](https://arxiv.org/pdf/2008.04277.pdf): Overview of Sentiment Analysis of
Code-Mixed Tweets" first and then experiment to study the effects of Self-Training keeping these models as the backbone. The results observed were as follows:

|Model     | Accuracy (%)     |
| ------------- | ------------- |
| Baseline mBERT  | 66         |
| ST mBERT |  **68.6** |
|Baseline RoBERTa-Base |  63|
|ST RoBERTa-Base | 68.3|
|Baseline XLM-RoBERTa| 66.9 |
|ST XLM-RoBERTa |  68.3|

#### ST denotes self-training and baseline denotes the pre-train model finetuning without self-training.

**Sentiment_Analysis** folder contains the following: <br />
 src -  contains the codes needed for running self-training and baseline experiments, along with the code to pre-process the data <br />
 data - contains the original data from the SemEval task and the data we used in the baseline and self-training experiments <br />
 logs-  terminal outputs of the experiments we ran <br />
 
 To run the code, make sure the following libraries are installed:(Other libraries are some standard packages that come with general Anaconda installations)
 
 ```
pip install torch==1.5.0 
pip install torchtext==0.6.0
pip install git+https://github.com/huggingface/transformers.git
```
 
 **Running the code**
  ```
cd Sentiment_Analysis/src
python self_train_XLMRoberta_linear.py
python self_train_bert.py
python self_train_roberta.py
```

 
 ### Named Entity Recognition
 
 For NER, we first finetuned various pre-trained models directly on [IIITH](https://github.com/SilentFlame/Named-Entity-Recognition/tree/master/Twitterdata)'s NER data.
 And then tried self-training for the models XLM-RoBERTa-Large and BERT-Large. <br />
 
 **NER** directory contains the following: <br />
 data - contains the csv files for the baseline and self-train experiments. <br />
 src - codes for the baseline and self-train experiments. also contains the preprocessing codes we used for converting the labeled and unlabeled data to the required formats. <br />
 
 The baseline results are:
 
 |Model     | F1 Score(%)    |
| ------------- | ------------- |
| BERT-Base  | 76.91        |
| RoBERTa-Base |  76.39 |
| mBERT-Base|  78.15|
|BERT-Large| 79.53|
|XLM-RoBERTa-Large| **80.6** |

The self-train results are:
|Model |F1-score(%) |
|---------| ------|
|BERT-Large| 80.76 |
| XLM-RoBERTa-Large | **81.24** |
 
 For running the codes:
 
   ```
cd NER/src
pip install simpletransformers
python ST_BERT_large_cased_NER.py
python Bert_large_linear.py
```
 (And similarly for other files)
 
 
 
 
 
