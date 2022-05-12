import pandas as pd, re, string, time
import requests
word_list =[]
file_names = {'test': ['Hinglish_test_unalbelled_conll_updated.txt', 'hinglish_test_unt_text.txt']}
f1 = open("/content/drive/My Drive/CS769/Semeval_2020_task9_data/Hinglish/" + file_names['test'][0],"r")
f_label = open("/content/drive/My Drive/CS769/Semeval_2020_task9_data/Hinglish/Hinglish_test_labels.txt","r")
f = open("/content/drive/My Drive/CS769/BAKSA_IITK/data/" + file_names['test'][1],"a+")


f.write('uid\ttext\tlabel\n')

sentiment_label=0
sentence_id=0
line =f1.readline()

df = pd.read_csv("/content/drive/My Drive/CS769/Semeval_2020_task9_data/Hinglish/Hinglish_test_labels.txt", header = None, skiprows=[0])
L = df.columns.values.tolist()

C1 = df[L[0]].tolist()#print(label[0])
C2 = df[L[1]].tolist()

labels = {}
for i in range(len(C1)):
    if C1[i] not in labels.keys():
        # print(C1[i])
        # uik
        labels[C1[i]] = C2[i]
Lkeys = list(labels.keys())


def remove_emojis(data): # from stackoverflow
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
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
    R = re.sub('  ', '', R)

    return R



transliterated_word_list=[]
transliterated_sentence = ""
hindi_list =[]

count = 0
k = 0
print('hoi3ogb')
hindi_translt = False
uid_numadd = 700000
while(line):

    if(line=="\n"):
        if(len(word_list)):
            count+=1

            transliterated_word_list += [str(sentence_id), "\t"] + word_list[:-1]
            transliterated_word_list += ["\t", str(sentiment_label), "\t", str(sentiment_label), "\n" ]
            if(count%1==0):
                dummy_list=[]
                for elem in hindi_list:
                    if len(elem)>8:
                        dummy_list.append(elem[:8])
                    else:
                        dummy_list.append(elem)
                hindi_list = dummy_list
                hindi_string = " ".join(hindi_list)
                if hindi_translt == True:
                  URL = "https://www.google.com/inputtools/request?text="+str(hindi_string)+"&ime=transliteration_en_hi&num=5&cp=0&cs=0&ie=utf-8&oe=utf-8&app=jsapi&uv"
                  PARAMS = {} 
                  r = requests.get(url = URL, params = PARAMS) 
                  data = r.json() 
                  time.sleep(0.001)
                  hindi_translation = data[1][0][1][0]

                  hindi_translation_list = hindi_translation.split(" ")
                else:
                  hindi_translation_list = hindi_string.split(' ')

                j=0
                for i in range(len(transliterated_word_list)):
                    if(transliterated_word_list[i]=="qwertyuiopasdfghjklzxcvbnm"):
                        transliterated_word_list[i] = hindi_translation_list[j]
                        j+=1
                transliterated_word_list_copy = transliterated_word_list.copy()
                transliterated_sentence = "".join(transliterated_word_list)
                transliterated_sentence_copy = "".join(transliterated_word_list)
                li=re.findall(r'@\s[a-zA-Z0-9]*\s',transliterated_sentence)
                lis = []
                for elems in li:
                  a = elems.split()
                  lis.extend(a)
                ref_sent = [elem for elem in transliterated_word_list_copy if elem not in lis]

                R = ' '.join(ref_sent)

                rem_num = re.sub(r'\d+', '', R)
                rem_punc = "".join([char for char in rem_num if char not in string.punctuation])
                rem_dspace = re.sub('\s+', ' ', rem_punc).strip()
                R1 = re.sub(r'…', '', rem_dspace)
                R1=re.sub(r"\shttp.*", "", R1)
                R1 = re.sub(r'˜', '', R1)
                R1 = re.sub(r'[^\w\s]', '', R1)
                s_id = int(re.sub(r'\n', '', sentence_id))

                transliterated_sentence = str(s_id+uid_numadd) + '\t' + R1[0:-3] + '\t' + str(sentiment_label) + '\n'
                if 'RT' in transliterated_sentence:
                  transliterated_sentence = transliterated_sentence.replace('RT ', '')
                  transliterated_sentence = transliterated_sentence.replace(' RT ', '')
                transliterated_sentence = remove_emojis(transliterated_sentence)
                if len(transliterated_sentence.split()) > 5:
                  f.write(transliterated_sentence)
                transliterated_word_list=[]
                hindi_list =[]
            word_list=[]

    else:
        array = line.split("\t")
        if(array[0]=='meta'):
            sentence_id = array[1]

            sentiment_label_map = {"negative":0, "positive":2, "neutral":1}
            sentiment_label = sentiment_label_map[labels[int(array[1][:-1])]]
        else:
            if(array[1][:-1]=="Hin"):
                hindi_list.append(array[0])
                word_list += ["qwertyuiopasdfghjklzxcvbnm", " "]
            else:
                word_list += [array[0], " "]
    line = f1.readline()

if len(transliterated_sentence.split()) > 5:
  f.write(transliterated_sentence)
