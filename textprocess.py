# -*- coding: utf-8 -*-
import csv
import jieba
import codecs
from langconv import * # convert Traditional Chinese characters to Simplified Chinese characters
import random
import re
datafile=r'C:\Users\guan\Desktop\data\bookcomment.csv'#数据源文件
Xfile=r'C:\Users\guan\Desktop\data\X.txt'#分词后的语句文件
Yfile=r'C:\Users\guan\Desktop\data\Y.txt'#标注文件，与语句文件逐行对应
stopfile=r'C:\Users\guan\Desktop\data\chinese_stop_words.txt'#停止词文件
stopwords=[line.rstrip() for line in codecs.open(stopfile,'r',encoding="utf-8")]
data=[]
with codecs.open(datafile, "r") as doc:
    file=csv.reader(doc)
    for line in file:
        try:
            starpre=line[6]
            if starpre=='力荐':
                star=2
            elif starpre=='推荐':
                star=1
            elif starpre=='还行':
                star=0
            elif starpre=='较差':
                star=0
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
            for seg in seglist:#此循环去掉了停止词和空词，若不想去掉则改动以下数行
                if seg not in stopwords:
                    if seg != ' ':
                        final.append(seg)
            if len(final)>0:
                data.append((final, star))
        except:
            continue

print (len(data))
with codecs.open(Xfile,'w',encoding='UTF-8') as out:
    for line in data:
        s=''
        for item in line[0]:
            s=s+str(item)+' '
        out.write(s+'\r\n')
with codecs.open(Yfile,'w',encoding='UTF-8') as out:
    for line in data:
        out.write(str(line[1])+'\r\n')
