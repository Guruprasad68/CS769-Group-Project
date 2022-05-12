#Mounting the drive
from google.colab import drive
drive.mount('/content/drive')

#Importing the necessary libraries
import pandas as pd
import seqeval
import re

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