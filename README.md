# CS769-Group-Project
This repository is for Semester long project (Assignment 3 and 4 mainly) of UW Madison's Spring 22 course CS769: Advanced Natural Language Processing by Dr. Junjie Hu.

Topic : Sentiment Analyisis and Named Entity Recognition of Code-Switched Hinglish(Hindi+English) Social Media Data

Group Members: [Guruprasad](https://github.com/Guruprasad68),  [Nitin](https://github.com/nitinimage), [Siddharth](https://github.com/sidhsmani)

**Assignment_3** Folder contains re-implementation of "Kumar et al., BAKSA at SemEval-2020 Task 9: Bolstering CNN with Self-Attention for Sentiment Analysis of Code Mixed Text".<br />
Link to their [code](https://github.com/keshav22bansal/BAKSA_IITK) and [paper](https://arxiv.org/pdf/2007.10819.pdf).<br />
Apart from re-implementing their work, we also implemented their idea on other pre-trained models [mT5](https://arxiv.org/pdf/2010.11934.pdf), [IndicBERT](https://indicnlp.ai4bharat.org/papers/arxiv2020_indicnlp_corpus.pdf) available on [HuggingFace](https://huggingface.co/) and studied the performance of removing the ensemble structure and replacing it with just a linear layer with a tunable encoder.

**Assignment_4** Folder contains continuation of work on Sentiment Analysis from the last assignment using Self-Training. We also study Hindi-English code-switched NER using [IIITH](https://github.com/SilentFlame/Named-Entity-Recognition/tree/master/Twitterdata)'s dataset. We first finetune different pre-trained models of the BERT family on this dataset using just a linear layer, and later explore the idea of Self-Training for NER. The [XLM-RoBERTa](https://arxiv.org/abs/1911.02116) and [Bert-Large](https://arxiv.org/pdf/1810.04805.pdf) models gave us improvement on the F1 score.

For both the tasks, we use unlabeled data released by the authors of [HinglishNLP at SemEval-2020 Task 9](https://aclanthology.org/2020.semeval-1.119/).
For Assignment 4, we use some parts of code and finetuned models from https://github.com/NirantK/Hinglish


**Note:** Assignment 3 codes related to finetuning with a linear layer, pre-processing data were written by us and rest were modified from https://github.com/keshav22bansal/BAKSA_IITK.

More detailed information about Assignment 3 can be found in the [report](https://drive.google.com/file/d/17yPAq8MD6m2nbfHpXC-YWdSe_PWX56tm/view?usp=sharing) and Assignment 4 can be found in the [report](https://drive.google.com/file/d/1zbWyti9VyAQbq0QIMq1fgd2zKsU7l_CG/view?usp=sharing) 

**Detailed README for the assignments in their respective folders.**
