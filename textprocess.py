# -*- coding: utf-8 -*-
import csv
import jieba
import codecs
from langconv import *
import random
import re
from gensim.models import Word2Vec
datafile=r'C:\Users\guan\Desktop\data\comment.csv'#file of resources
Xfile=r'C:\Users\guan\Desktop\data\X_2and5.txt'#file of inputs sentences
Yfile=r'C:\Users\guan\Desktop\data\Y_2and5.txt'#file of labels
stopfile=r'C:\Users\guan\Desktop\data\chinese_stop_words.txt'#file of stop words
word2vec=r'C:\Users\guan\Desktop\data\word2vec_32.model'

def main():
    #creatdataset([1,4])
    creatword2vec(32)

stopwords=[line.rstrip() for line in codecs.open(stopfile,'r',encoding="utf-8")]
data=[[],[],[],[],[]]
data_word2vec=[]
with codecs.open(datafile, "r",encoding='utf-8') as doc:
    file=csv.reader(doc)
    for line in file:
        try:
            #Obtain the sentence of the classes I need
            starpre=line[0]
            if starpre=='力荐':
                star=4
            elif starpre=='推荐':
                star=3
            elif starpre=='还行':
                star=2
            elif starpre=='较差':
                star=1
            elif starpre=='很差':
                star=0
            else :
                continue
            text=line[1]
            #only chinese character can be saved
            p = re.compile(r'[^\u4e00-\u9fa5]')
            text = " ".join(p.split(text)).strip()
            #convert Traditional Chinese characters to Simplified Chinese characters
            text = Converter('zh-hans').convert(text)
            #abandon the line breaker
            text = text.replace("\n", "")
            text = text.replace("\r", "")
            #cut the sentence into words(perhaps phrase?)
            seglist=jieba.cut(text,cut_all=False)
            seglist=list(seglist)
            final=[]
            for seg in seglist:#abandon stop words
 #               if seg not in stopwords:#abandon empty lines,when training word2vec ,#this line
                if seg != ' ':
                    final.append(seg)
            if len(final)>0:
                #for convenience in shuffle action,I use a tuple to keep the pair.
  #              data[star].append((final,star))#abandon this line when training word2vec
                data_word2vec.append(final)
        except:
            continue
#if I should extract part of the sentence,I will shuffle them first,because the sentence in the file is ranked in some order.
def creatdataset(starlist):
    data_final=[]
    #extract the data to one list
    for i in starlist:
        data_final.extend(data[i])
    print (len(data_final))
    #save the sentences and labels separately 
    with codecs.open(Xfile,'w',encoding='UTF-8') as out:
        for line in data_final:
            s=''
            for item in line[0]:
                s=s+str(item)+' '
            out.write(s+'\r\n')
    with codecs.open(Yfile,'w',encoding='UTF-8') as out:
        for line in data_final:
            out.write(str(line[1])+'\r\n')
#The following lines train a word2vec model
def creatword2vec(dimention):    
    random.shuffle(data_word2vec)
    print(len(data_word2vec))
    model=Word2Vec(data_word2vec,size=dimention)
    model.save(word2vec)
if __name__ =='__main__':
    main()
