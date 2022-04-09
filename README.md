# CS769-Group-Project
This repository is for Assignment 3 and 4 of UW Madison's Spring 22 course CS769: Advanced Natural Language Processing.

Group Members: [Guruprasad](https://github.com/Guruprasad68),  [Nitin](https://github.com/nitinimage), [Siddharth](https://github.com/sidhsmani)


Topic : Hinglish(Hindi+English) SentiMent Analyisis in Code Switching scenario

Install these dependencies:


pip install transformers\
pip install torch==1.5.0\
pip install torchtext==0.6.0


While using different models, need to change 'hidden_size' in this line appropriately.


embedding_dim = bert.config.to_dict()['hidden_size']





