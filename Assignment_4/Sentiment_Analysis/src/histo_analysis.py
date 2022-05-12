import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive') 
%cd /content/drive/MyDrive/769_Project_Assignment_4

# Test

def test_analysis(testoutput,testfile, n_bins):
  testoutput = testoutput.drop("Unnamed: 0", axis=1)
  testoutput.rename({'sentiment': 'prediction'}, axis=1, inplace=True)
  testoutput['uid'] = testoutput['uid'] - 700000

  testlabels = pd.read_csv("SA_Outputs/test_labels.csv")
  testlabels.rename({'sentiment': 'label'}, axis=1, inplace=True)
  testlabels['uid'] = testlabels['uid'] - 700000

  df = pd.merge(testoutput, testlabels, on='uid')
  df['flag'] = df['prediction'] == df['label']
  df['flag'] = df['flag'].astype(int)


  test_result = {}
  with open(testfile) as f:
    lines = f.readlines()
    for line in lines:
      if "meta" in line:
        id = line.split()[-1]
        test_result[id] = {"Eng":0, "Hin":0}
      if "Eng" in line:
        test_result[id]["Eng"]+=1
      if "Hin" in line:
        test_result[id]["Hin"]+=1


  df_mix = pd.DataFrame.from_dict(test_result, orient='index')
  df_mix.reset_index(level=0, inplace=True)
  df_mix.rename({'index': 'uid'}, axis=1, inplace=True)
  df_mix['uid'] = pd.to_numeric(df_mix['uid'])

  df2 = pd.merge(df, df_mix, on='uid')
  df2['per_hin'] = df2.apply(lambda row : 
                      100*row['Hin']/(row['Hin']+row['Eng']), axis = 1).round(2)

  finaldf = df2[['flag', 'per_hin']]
  bins = np.linspace(0, 100, n_bins)
  finaldf['bins'] = pd.cut(finaldf['per_hin'],bins)
  return finaldf


# Valid 

def valid_analysis(outfile,validfile, n_bins):
  output = pd.read_csv(outfile)
  output = output.drop("Unnamed: 0", axis=1)
  output.rename({'sentiment': 'prediction'}, axis=1, inplace=True)
  output['uid'] = output['uid'] - 700000


  valid_result = {}
  with open(validfile) as f:
    lines = f.readlines()
    for line in lines:
      if "meta" in line:
        label = line.split()[-1]
        id = line.split()[-2]
        valid_result[id] = {"label":label,"Eng":0, "Hin":0}
      if "Eng" in line:
        valid_result[id]["Eng"]+=1
      if "Hin" in line:
        valid_result[id]["Hin"]+=1

  df = pd.DataFrame.from_dict(valid_result, orient='index')
  df.reset_index(level=0, inplace=True)
  df.rename({'index': 'uid'}, axis=1, inplace=True)
  df.drop(921, inplace=True)
  df['uid'] = pd.to_numeric(df['uid'])


  df2 = pd.merge(df, output, on='uid')
  df2['flag'] = df2['label']==df2['prediction']
  df2['flag'] = df2['flag'].astype(int)
  df2['per_hin'] = df2.apply(lambda row : 
                       100*row['Hin']/(row['Hin']+row['Eng']), axis = 1).round(2)
  

  finaldf = df2[['flag', 'per_hin']]
  bins = np.linspace(0, 100, n_bins)
  finaldf['bins'] = pd.cut(finaldf['per_hin'],bins)

  return finaldf

testoutput = pd.read_csv("SA_Outputs/bert-test-output-df.csv")
testfile = "data/Semeval_2020_task9_data/Hinglish_test_unalbelled_conll_updated.txt"
n_bins = 31

test_df = test_analysis(testoutput,testfile, n_bins)

outfile = "SA_Outputs/bert-valid-output-df.csv"
validfile = "data/Semeval_2020_task9_data/Hinglish_dev_3k_split_conll.txt"

valid_df = valid_analysis(outfile, validfile, n_bins)


# df_combined = pd.concat([valid_df,test_df])
# bins = np.linspace(0, 100, n_bins)
# df_combined['bins'] = pd.cut(df_combined['per_hin'],bins)

import seaborn as sns

def plotdf(finaldf,title, n_bins):

  d = finaldf.groupby('bins').mean()
  ax = sns.regplot(x=d['per_hin'], y=d['flag'], ci=85)
  ax.set_xlabel("Percentage of Hindi Words")
  ax.set_ylabel("Accuracy")
  ax.set_ylim(0.5,0.8)
  ax.set_title(title)
  plt.show()

  c = finaldf[finaldf['flag']==1]['per_hin']
  ic = finaldf[finaldf['flag']==0]['per_hin']


  bins = np.linspace(0, 100, n_bins)

  plt.hist(c, bins, alpha=0.5, label='Correct Prediction')
  plt.hist(ic, bins, alpha=0.6, label='Incorrect Prediction')
  plt.xlabel('Percentage of Hindi Words')
  plt.title(title)
  plt.legend(loc='upper left')
  plt.show()
  plotdf(test_df,title="Code-Switching Impact Analysis on Test Accuracy", n_bins=n_bins)
  plotdf(valid_df,title="Code-Switching Impact Analysis on Valid Accuracy", n_bins=n_bins)