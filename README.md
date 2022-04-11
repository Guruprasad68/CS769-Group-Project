# CS769-Group-Project
This repository is for Assignment 3 and 4 of UW Madison's Spring 22 course CS769: Advanced Natural Language Processing by Dr. Junjie Hu.

Topic : Sentiment Analyisis and Named Entity Recognition of Code-Switched Hinglish(Hindi+English) Social Media Data

Group Members: [Guruprasad](https://github.com/Guruprasad68),  [Nitin](https://github.com/nitinimage), [Siddharth](https://github.com/sidhsmani)

**Assignment_3** Folder contains re-implementation of "Kumar et al., BAKSA at SemEval-2020 Task 9: Bolstering CNN with Self-Attention for Sentiment Analysis of Code Mixed Text".<br />
Link to their [code](https://github.com/keshav22bansal/BAKSA_IITK) and [paper](https://arxiv.org/pdf/2007.10819.pdf).<br />
Apart from re-implementing their work, we also implemented their idea on other pre-trained models [mT5](https://arxiv.org/pdf/2010.11934.pdf), [IndicBERT](https://indicnlp.ai4bharat.org/papers/arxiv2020_indicnlp_corpus.pdf) available on [HuggingFace](https://huggingface.co/) and studied the performance of removing the ensemble structure and replacing it with just a linear layer with a tunable encoder.

**Assignment_4** Folder will be updated soon, and will contain our idea of levaraging self-training to boost the performance of existing models and extend the same to Named Entity Recognition of Code-Switched Hinglish Data.


**Note:** Assignment 3 codes related to finetuning with a linear layer, pre-processing data were written by us and rest were modified from https://github.com/keshav22bansal/BAKSA_IITK.

More detailed information about Assignment 3 can be found in our [report](https://drive.google.com/drive/u/0/my-drive)
