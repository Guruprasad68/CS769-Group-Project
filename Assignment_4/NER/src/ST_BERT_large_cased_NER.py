#Mounting the drive
from google.colab import drive
drive.mount('/content/drive')

#Importing the necessary libraries
from simpletransformers.ner import NERModel, NERArgs
from simpletransformers.classification import ClassificationModel
import pandas as pd
import seqeval
import re

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

model=NERModel("bert","bert-large-cased", args=model_args)

#Training the model(Finetuning)

model.train_model(train_data=train_df,show_running_loss=True, eval_data= valid_df,acc=seqeval.metrics.accuracy_score)

#Testing the finetuned model on the test dataset

test_df=test_df.dropna()
result, model_outputs, wrong_preds=model.eval_model(test_df)
print(result)


f=open('/content/drive/My Drive/CS769/hw4/769_Project_Assignment_4/unused data/hinglish_unsup_concat4.txt')
lines=f.readlines()
# len(lines)
tweet=[]
tweets=[]
for i,line in enumerate(lines):
  if line!='\n':
    line_split=line.split("||")
    if len(line_split)==0:
      line_split=[line]
    if len(tweet)==0:
      for splits in line_split[:-1]:
        tweets.append(splits.strip())
      tweet.append(line_split[-1].strip())
    else:
      tweet.append(line_split[0].strip())
      #print(tweet)
      tweets.append(tweet[0]+" "+ tweet[1])
      tweet=[]
      #print(tweet)
      if len(line_split)>2:
        for split in line_split[1:-1]:
          tweets.append(split.strip())
        tweet.append(line_split[-1].strip())
      elif len(line_split)==2:
        tweet.append(line_split[-1].strip())
      else:
        continue

def remove_emojis(data): # from stackoverflow
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    R = re.sub(emoj, '', data)
    return R

cleaned_tweets = []
for tweet in tweets:
  R = remove_emojis(tweet)
  R = re.sub(r'RT','',R)
  R = re.sub(r'_','',R)
  R = re.sub(r'…', '', R)
  R = R.strip()
  if len(R.split(' ')) > 5:
    cleaned_tweets.append(R)
ct=cleaned_tweets[0:50000]

predictions, raw_outputs=model.predict(ct)

import pandas as pd
sent_id=2044  #Pseudo data starts with 2045
main_list=[]
for sentence in predictions:
  sent_id+=1
  for word_tag in sentence:
    sub_list=[]
    sub_list=[sent_id]
    word=list(word_tag.keys())[0]
    tag=word_tag[word]
    sub_list.append(str(word))
    sub_list.append(str(tag))
    main_list.append(sub_list)
df=pd.DataFrame(main_list,columns=['sentence_id','words','labels'])
df.to_csv('/content/drive/MyDrive/CS769/hw4/769_Project_Assignment_4/Named-Entity-Recognition/data/pseudo_bert_large_cased_50k.csv',index=False)
# concatenating pseudo data to labeled data
pseudo_df=pd.read_csv('/content/drive/MyDrive/CS769/hw4/769_Project_Assignment_4/Named-Entity-Recognition/data/pseudo_bert_large_cased_50k.csv')
concat_df=pd.concat([train_df,pseudo_df],ignore_index=True)
concat_df.to_csv('/content/drive/MyDrive/CS769/hw4/769_Project_Assignment_4/Named-Entity-Recognition/data/concat_bert_large_cased_50k.csv',index=False)
concat_df=concat_df.dropna()

model_args=NERArgs()
model_args.labels_list=labels
model_args.num_train_epochs=3
model_args.learning_rate=3e-5
model_args.train_batch_size=16
model_args.eval_batch_size=32
model_args.classification_report=True
model_args.overwrite_output_dir =True
model_args.evaluate_during_training=True
model_args.evaluate_during_training_verbose=True
model_args.evaluate_during_training_steps=10000
model_args.save_eval_checkpoints=False
model_args.save_model_every_epoch=False
model_args.scheduler='cosine_schedule_with_warmup'
# training student model

model=NERModel("bert","bert-large-cased",args=model_args)

#1:Training the model (#3 epochs)(#XLM_Roberta-Large after adding the psuedo labeled data)
model.train_model(train_data=concat_df,show_running_loss=True, eval_data= valid_df,acc=seqeval.metrics.accuracy_score)
test_df=test_df.dropna()
result,_,_=model.eval_model(test_df)
print(result)