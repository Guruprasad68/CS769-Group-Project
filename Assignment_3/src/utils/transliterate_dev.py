import pandas as pd, re, string, time
import requests
word_list =[]
f1 = open("/content/drive/My Drive/CS769/Semeval_2020_task9_data/Hinglish/Hinglish_dev_3k_split_conll.txt", "r")#Hinglish_dev_3k_split_conll.txt","r")
f_label = open("/content/drive/My Drive/CS769/Semeval_2020_task9_data/Hinglish/Hinglish_test_labels.txt","r")
f = open("/content/drive/My Drive/CS769/BAKSA_IITK/data/hinglish_dev_text_final.txt","a+")

f.write('uid\ttext\tlabel\n')

sentiment_label=0
sentence_id=0
line =f1.readline()
# print(line)
# uik
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
# print(type(Lkeys[0]))
# uik

transliterated_word_list=[]
transliterated_sentence = ""
hindi_list =[]

count = 0
k = 0
print('hoi3ogb')
while(line):
    # k += 1
    # print(k)
    if(line=="\n"):
        if(len(word_list)):
            count+=1
            # print(sentence_id, sentiment_label)
            # uik
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
                URL = "https://www.google.com/inputtools/request?text="+str(hindi_string)+"&ime=transliteration_en_hi&num=5&cp=0&cs=0&ie=utf-8&oe=utf-8&app=jsapi&uv"
                PARAMS = {} 
                r = requests.get(url = URL, params = PARAMS) 
                data = r.json() 
                time.sleep(0.001)
                hindi_translation = data[1][0][1][0]
                # print('info:', hindi_translation)
                # gfwg3
                hindi_translation_list = hindi_translation.split(" ")

                j=0
                for i in range(len(transliterated_word_list)):
                    if(transliterated_word_list[i]=="qwertyuiopasdfghjklzxcvbnm"):
                        transliterated_word_list[i] = hindi_translation_list[j]
                        j+=1
                transliterated_word_list_copy = transliterated_word_list.copy()
                # uik
                transliterated_sentence = "".join(transliterated_word_list)
                # print(transliterated_sentence, type(transliterated_sentence))
                # print(transliterated_sentence)
                # uik
                transliterated_sentence_copy = "".join(transliterated_word_list)
                # lts = list(transliterated_sentence); 
                li=re.findall(r'@\s[a-zA-Z0-9]*\s',transliterated_sentence)
                lis = []
                for elems in li:
                  a = elems.split()
                  # print(a)
                  lis.extend(a)
                # print(lis, transliterated_sentence)
                # uik
                # print(transliterated_word_list_copy)
                # uik
                # transliterated_word_list_copy.remove()
                ref_sent = [elem for elem in transliterated_word_list_copy if elem not in lis]
                # ref_sent.append('\n')
                # print(ref_sent, ' '.join(ref_sent))
                # uik
                R = ' '.join(ref_sent)
                # print(R)
                # uik
                rem_num = re.sub(r'\d+', '', R)
                rem_punc = "".join([char for char in rem_num if char not in string.punctuation])
                rem_dspace = re.sub('\s+', ' ', rem_punc).strip()
                # print(rem_punc, rem_dspace)
                # uik
                R1 = re.sub(r'…', '', rem_dspace)
                R1=re.sub(r"\shttp.*", "", R1)
                R1 = re.sub(r'˜', '', R1)
                # print('info:', R1, list(sentence_id), R1[-4])
                s_id = re.sub(r'\n', '', sentence_id)
                # uik
                transliterated_sentence = str(s_id) + '\t' + R1[0:-3] + '\t' + str(sentiment_label) + '\n'

                # print('Finally:', transliterated_sentence)#, '\n', list(transliterated_sentence))#, '\n', transliterated_sentence)#, list(transliterated_sentence))
                k += 1
                # uik
                # transliterated_sentence = list(transliterated_sentence)
                # ref_sent = [ elem for elem in li if elem % 2 != 0]
                # print(hindi_translation, transliterated_sentence, transliterated_word_list)
                # print(type(transliterated_sentence))
                # k += 1
                # if k == 1:
                  # iol
                # print(transliterated_sentence, list(transliterated_sentence), type(transliterated_sentence))
                # uik
                f.write(transliterated_sentence)
                transliterated_word_list=[]
                hindi_list =[]
            word_list=[]

    else:
        array = line.split("\t")
        if(array[0]=='meta'):
            sentence_id = array[1]
            # print(array)
            # err
            # print('HERE:', array[2][:-1], type(array[1][:-1]), array[1][:-1])
            sentiment_label_map = {"negative":0, "positive":2, "neutral":1}
            sentiment_label = sentiment_label_map[array[2][:-1]]
            # print(sentiment_label)
            # uiio
        else:
            if(array[1][:-1]=="Hin"):
                hindi_list.append(array[0])
                word_list += ["qwertyuiopasdfghjklzxcvbnm", " "]
            else:
                word_list += [array[0], " "]
    line = f1.readline()
    # print(line)
    # uik
    # if k == 3:
    #   # uik
    #   break

# print(transliterated_sentence)
# uik
f.write(transliterated_sentence)
