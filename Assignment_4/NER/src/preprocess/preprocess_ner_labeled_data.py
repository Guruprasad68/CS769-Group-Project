#importing drive
from google.colab import drive
drive.mount('/content/drive')

#Many different steps are involved because we tried NER using different approaches, finally only used
#csv files needed for simpletransformers library(train_new.csv, valid_new.csv, test_new.csv)

#Importing necessary libraries
import pandas as pd
import numpy as np
import re
import json

#Processing the annotated.csv file(original/IIITH data)
labels=pd.read_csv('/content/Named-Entity-Recognition/Twitterdata/annotatedData.csv')
with open('tweets.txt','w') as f, open('annotations.txt','w') as g:
    for index,row in labels.iterrows():
      #writing the tweets to tweets.txt
      if type(row['Sent'])==np.float:
        if np.isnan(row['Sent']):
          f.write('\n')
      else:
        if row['Word']!=np.float:
          f.write(str(row['Word'])+' ')
        else:
          pass

      #Writing the labels to annotations.txt

      if type(row['Sent'])==np.float:
        if np.isnan(row['Sent']):
          g.write('\n')
      else:
        g.write(row['Tag']+' ')


f1=open('train_tweets.txt','w')
f2=open('train_annotations.txt','w')
f3=open('valid_tweets.txt','w')
f4=open('valid_annotations.txt','w')
f5=open('test_tweets.txt','w')
f6=open('test_annotations.txt','w')

with open('annotations.txt') as ner , open('tweets.txt') as twe:
  lines2=ner.readlines()
  lines1=twe.readlines()
  k=0
  for i in range(3084):
    if len(lines1[i].split())==len(lines2[i].split()):
      if k>=0 and k<8:
        f1.write(lines1[i])
        f2.write(lines2[i])
        k+=1
      elif k>=8 and k<10:
        f3.write(lines1[i])
        f4.write(lines2[i])
        k+=1
      elif k==10:
        f5.write(lines1[i])
        f6.write(lines2[i])
        k+=1
      else:
        k=0
        f5.write(lines1[i])
        f6.write(lines2[i])


def create_json_csv(tweet_file,ner_file,set):
    tweet=open(tweet_file).read().split('\n')
    ner=open(ner_file).read().split('\n')

    raw_data={'sentence_id':[i for i in range(len(ner))],
              'words': [line for line in tweet],
                'tag': [line for line in ner]}
  

    df=pd.DataFrame(raw_data, columns=['sentence_id','words','tag'])
    df.to_json(set+'_new.json',orient='records',lines=True)
    df.to_csv(set+'_new.csv',index=False)

create_json_csv('/content/drive/MyDrive/769_Project_Assignment_4/Named-Entity-Recognition/data/helper_files/train_tweets.txt',
            '/content/drive/MyDrive/769_Project_Assignment_4/Named-Entity-Recognition/data/helper_files/train_annotations.txt',
            set='train')
create_json_csv('/content/drive/MyDrive/769_Project_Assignment_4/Named-Entity-Recognition/data/helper_files/valid_tweets.txt',
            '/content/drive/MyDrive/769_Project_Assignment_4/Named-Entity-Recognition/data/helper_files/valid_annotations.txt',
            set='valid')
create_json_csv('/content/drive/MyDrive/769_Project_Assignment_4/Named-Entity-Recognition/data/helper_files/test_tweets.txt',
            '/content/drive/MyDrive/769_Project_Assignment_4/Named-Entity-Recognition/data/helper_files/test_annotations.txt',
            set='test')


#converting _new.jsons to the required format of simpletransformers

def convert_new_json_to_new_csv(jsonpath,split='train'):
  lines=open(jsonpath).readlines()
  df=pd.DataFrame(columns=['sentence_id','words','labels'])
  for line in lines:
    line_dict=json.loads(line)
    sentence=line_dict['words'].split()
    tags=line_dict['tag'].split()
    for word,tag in zip(sentence,tags):
      df=df.append({'sentence_id':line_dict['sentence_id'],'words':word,'labels':tag},ignore_index=True)
  df.to_csv(split+'_new.csv',index=False)


convert_new_json_to_new_csv('/content/drive/MyDrive/769_Project_Assignment_4/Named-Entity-Recognition/data/train_new.json','train')
convert_new_json_to_new_csv('/content/drive/MyDrive/769_Project_Assignment_4/Named-Entity-Recognition/data/valid_new.json','valid')
convert_new_json_to_new_csv('/content/drive/MyDrive/769_Project_Assignment_4/Named-Entity-Recognition/data/test_new.json','test')