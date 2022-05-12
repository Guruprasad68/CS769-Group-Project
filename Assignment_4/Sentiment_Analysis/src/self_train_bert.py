from google.colab import drive
drive.mount('/content/drive') 
%cd /content/drive/MyDrive/769_Project_Nitin/Hinglish


!pip install fastcore==1.0.13
!pip install transformers==3.3.1
!pip install wandb==0.10.5

import wandb
wandb.login()

from hinglish import HinglishTrainer

hinglishbert = HinglishTrainer(
    model_name = "bert",
    batch_size = 32,
    attention_probs_dropout_prob = 0.4,
    learning_rate = 1e-6,
    adam_epsilon = 1e-8,
    hidden_dropout_prob = 0.3,
    epochs = 5,
    lm_model_dir = "bert",
    wname="bert",
    drivepath="repro",
    train_json="self_train_data/concat_train.json",
    dev_json="self_train_data/valid.json",
    test_json="self_train_data/test.json",
    test_labels_csv="self_train_data/test_labels.csv",
    )

hinglishbert.train()
hinglishbert.evaluate()