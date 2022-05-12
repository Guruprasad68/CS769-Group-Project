#Mounting the drive
from google.colab import drive
drive.mount('/content/drive')

#Importing the necessary libraries
from simpletransformers.ner import NERModel, NERArgs
from simpletransformers.classification import ClassificationModel
import pandas as pd
import seqeval

#Loading the datasets
#loading train, test, validation data
train_df=pd.read_csv('/content/drive/MyDrive/769_Project_Assignment_4/Named-Entity-Recognition/data/train_new.csv')
valid_df=pd.read_csv('/content/drive/MyDrive/769_Project_Assignment_4/Named-Entity-Recognition/data/valid_new.csv')
test_df=pd.read_csv('/content/drive/MyDrive/769_Project_Assignment_4/Named-Entity-Recognition/data/test_new.csv')


#Getting the labels in a list
labels=train_df['labels'].unique().tolist()


#Defining the arguments for the model

model_args=NERArgs()
model_args.labels_list=labels
model_args.num_train_epochs=20
model_args.learning_rate=3e-5
model_args.train_batch_size=16
model_args.eval_batch_size=32
model_args.classification_report=True
model_args.overwrite_output_dir =True
model_args.evaluate_during_training=True
model_args.evaluate_during_training_verbose=True
model_args.evaluate_during_training_steps=64
model_args.save_eval_checkpoints=False
model_args.save_model_every_epoch=False
model_args.scheduler='cosine_schedule_with_warmup'


# Loading the pre-trained model

model=NERModel("bert", "bert-base-multilingual-cased", args=model_args)

#Training the model(Finetuning)

model.train_model(train_data=train_df,show_running_loss=True, eval_data= valid_df,acc=seqeval.metrics.accuracy_score)

#Testing the finetuned model on the test dataset

test_df=test_df.dropna()
result, model_outputs, wrong_preds=model.eval_model(test_df)
print(result)