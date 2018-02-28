# -*- coding: utf-8 -*-
import codecs
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.core import Dense
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.callbacks import TensorBoard
Xfile=r'C:\Users\guan\Desktop\data\X.txt'
Yfile=r'C:\Users\guan\Desktop\data\Y.txt'
Xf=codecs.open(Xfile,'r',encoding='UTF-8')
Yf=codecs.open(Yfile,'r',encoding='UTF-8')
X=[]
Y=[]
for line in Xf:
    X.append(line.strip('\r\n').strip(' ').split(' '))
for line in Yf:
    Y.append(line.strip())
Xf.close()
Yf.close()
length=len(Y)
X = [" ".join(wordslist) for wordslist in X]
tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

word_index = tokenizer.word_index
wordnum=len(word_index)
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=None)

labels = to_categorical(np.asarray(Y))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

def vocal_size():
    return len(word_index)
def gettotalX():
    return data
def gettotalY():
    return labels

splitnum=int(0.8*length)
x_train = data[:splitnum]
y_train = labels[:splitnum]
splitnum+=1
x_val = data[splitnum:]
y_val = labels[splitnum:]
print (len(x_train))
print (len(x_val))

def gettrainX():
    return x_train
def gettrainY():
    return y_train
def gettestX():
    return x_val
def gettestY():
    return y_val

