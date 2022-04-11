### Re-implementation of the code and extending the idea of Kumar et al.,

The data for the assignment comes from "[SemEval-2020 Task 9](https://arxiv.org/pdf/2008.04277.pdf): Overview of Sentiment Analysis of
Code-Mixed Tweets"

**Original Data** can be found in data/Semeval_2020_task9_data/Hinglish <br>
**Pre-processing** codes in src/utils/ <br>
**Pre-processed** data can be found in data/hinglish <br>
 **Terminal outputs** of the experiments did in this assignment can be found in Results/ <br>
**Code** can be found in src/ <br>
**Checkpoints** will be stored in checkpoint/ (Empty before you run code)


To try out the codes, make sure the following dependencies are there, else run the following commands in your terminal:

```
pip install torch==1.5.0 <br>
pip install torchtext==0.6.0 <br>
pip install git+https://github.com/huggingface/transformers.git
```

We advise using a virtual environment or a cloud platform like Google Colaboratory.

**Running the code:**
```
mkdir checkpoint <br>
cd src <br>
python IndicBERT_Ensemble.py hinglish <br>
python IndicBERT_Linear.py
```

Similarly for other models.

**Results obtained in our re-implementation and extension work** <br>

|Model     | Accuracy      |
| ------------- | ------------- |
| XLM-RoBERTa-Ensemble  | 65.14         |
| MT5-Ensemble |  63.61 |
|IndicBERT-Ensemble |  58.37|
|XLM-RoBERTa-Linear | **68.11**|
|MT5-Linear | 61 |
|IndicBERT-Linear |  61.54|
