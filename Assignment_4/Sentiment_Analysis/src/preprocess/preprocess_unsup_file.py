from google.colab import drive
drive.mount('/content/drive')
# %cd /content/drive/MyDrive/769_Project_Assignment_4/
import re
f=open('/content/drive/MyDrive/769_Project_Assignment_4/unused data/hinglish_unsup_concat4.txt')
lines=f.readlines()
# print(lines)

#Getting tweets in a one line each
tweet=[]
tweets=[]
for i,line in enumerate(lines):
  if line!='\n':
    #print(line)
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
# writing tweets to file in required format 
file=open('/content/drive/MyDrive/769_Project_Assignment_4/preprocessed_unlabeled_tweets.txt','w')
file.write('uid,text,label\n')
for i,tweet in enumerate(tweets):
  #basic preprocessing of tweets
  tweet=re.sub(r"\shttp.*", "", tweet)
  tweet=re.sub(r'…', '', tweet)
  tweet=re.sub(r'@[a-zA-Z0-9]*','',tweet)
  tweet=re.sub(r'RT','',tweet)
  tweet=re.sub(r'[^\w\s]', '', tweet)
  tweet=re.sub(r'_','',tweet)
  tweet=tweet.strip()
  if len(tweet.split())>5:
    file.write(str(i)+"," + tweet +"," + str(0) + "\n")
    #print(str(i)+"\t"+ tweet+"\t"+str(0)+"\n")