import codecs
from queue import Queue
import numpy as np
from gensim.models import KeyedVectors
def data():
    n_class=2
    len_max=50
    file=r"C:\Users\guan\Desktop\data\news12g_bdbk20g_nov90g_dim128\news12g_bdbk20g_nov90g_dim128.bin"
    Xfile=r'C:\Users\guan\Desktop\data\X.txt'
    Yfile=r'C:\Users\guan\Desktop\data\Y.txt'
    Xf=codecs.open(Xfile,'r',encoding='UTF-8')
    Yf=codecs.open(Yfile,'r',encoding='UTF-8')
    X=[]
    Y=Queue()
    for line in Xf:
        X.append(line.strip('\r\n').strip(' ').split(' '))
    for line in Yf:
        Y.put(int(line.strip()))
    Xf.close()
    Yf.close()

    word_vectors = KeyedVectors.load_word2vec_format(file, binary=True)
    X_embedding=[]
    Y_vec=[]
    vector_blank=np.array([0.0]*64)
    for sentence in X:
        sentence_embedding=[]
        for word in sentence :
            try:
                sentence_embedding.append(word_vectors.wv[word])
            except:
                continue
        index=Y.get()
        if len(sentence_embedding)>0 and len(sentence_embedding)<=len_max :
            for i in range(len_max-len(sentence_embedding)):
                sentence_embedding.append(vector_blank)
            X_embedding.append(np.array(sentence_embedding))
            label_vec=[0]*n_class
            label_vec[index]=1
            Y_vec.append(np.array(label_vec,dtype=int))
#    X_embedding=X_embedding[:140000]
 #   Y_vec=Y_vec[:140000]
    X_embedding=np.array(X_embedding)
    Y_vec=np.array(Y_vec)
    print(X_embedding.shape,Y_vec.shape)
    return X_embedding,Y_vec
