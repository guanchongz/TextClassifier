# -*- coding: utf-8 -*-
import csv
import jieba
import codecs
from langconv import * # convert Traditional Chinese characters to Simplified Chinese characters
import random
import re
#from gensim.models import Word2Vec
datafile=r'C:\Users\guan\Desktop\data\comment.csv'#file of resources
Xfile=r'C:\Users\guan\Desktop\data\X_3class.txt'#file of inputs sentences
Yfile=r'C:\Users\guan\Desktop\data\Y_3class.txt'#file of labels
stopfile=r'C:\Users\guan\Desktop\data\chinese_stop_words.txt'#file of stop words
word2vec=r'C:\Users\guan\Desktop\data\word2vec_full.model'
stopwords=[line.rstrip() for line in codecs.open(stopfile,'r',encoding="utf-8")]
data=[[],[],[],[],[]]
with codecs.open(datafile, "r",encoding='utf-8') as doc:
    file=csv.reader(doc)
    for line in file:
        try:
            starpre=line[0]
            if starpre=='力荐':
                star=2
            elif starpre=='推荐':
                star=3
            elif starpre=='还行':
                star=1
            elif starpre=='较差':
                star=4
            elif starpre=='很差':
                star=0
            else :
                continue
            text=line[1]

            p = re.compile(r'[^\u4e00-\u9fa5]')
            text = " ".join(p.split(text)).strip()

            text = Converter('zh-hans').convert(text)
            text = text.replace("\n", "")
            text = text.replace("\r", "")
            seglist=jieba.cut(text,cut_all=False)
            seglist=list(seglist)
            final=[]
            for seg in seglist:#abandon stop words
                if seg not in stopwords:
                    if seg != ' ':
                        final.append(seg)
            if len(final)>0:
                data[star].append((final,star))
        except:
            continue
#for i in range(5):
 #   random.shuffle(data[i])
data_final=[]
for i in range(3):
    data_final.extend(data[i][:30000])
print (len(data_final))
random.shuffle(data_final)
#data_final=data_final[:100000]
#model=Word2Vec(data_final,size=128)
#model.save(word2vec)
with codecs.open(Xfile,'w',encoding='UTF-8') as out:
    for line in data_final:
        s=''
        for item in line[0]:
            s=s+str(item)+' '
        out.write(s+'\r\n')
with codecs.open(Yfile,'w',encoding='UTF-8') as out:
    for line in data_final:
        out.write(str(line[1])+'\r\n')
