import codecs
from queue import Queue
import numpy as np
from gensim.models import Word2Vec
def main():
    n_class=3
    len_max=50
    file=r"C:\Users\guan\Desktop\data\news12g_bdbk20g_nov90g_dim128\news12g_bdbk20g_nov90g_dim128.bin"
    Xfile=r'C:\Users\guan\Desktop\data\X_3class.txt'
    Yfile=r'C:\Users\guan\Desktop\data\Y_3class.txt'
    X_embedding_file=r'C:\Users\guan\Desktop\data\X_embedding_self_3class'
    Y_vec_file=r'C:\Users\guan\Desktop\data\Y_vec_self_3class'
    length_sentence_file=r'C:\Users\guan\Desktop\data\length_3class'
    word2vec=r'C:\Users\guan\Desktop\data\word2vec_full.model'
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

    word_vectors = Word2Vec.load(word2vec)
    X_embedding=[]
    Y_vec=[]
    length_sentence=[]
    vector_blank=np.array([0.0]*128)
    for sentence in X:
        sentence_embedding=[]
        for word in sentence :
            try:
                sentence_embedding.append(word_vectors.wv[word])
            except:
                continue
        index=Y.get()
        length=len(sentence_embedding)
        if length>0 and length<=len_max :
            for i in range(len_max-length):
                sentence_embedding.append(vector_blank)
            X_embedding.append(np.array(sentence_embedding))
            label_vec=[0]*n_class
            label_vec[index]=1
            length_sentence.append(length)
            Y_vec.append(np.array(label_vec,dtype=int))
    X_embedding=np.array(X_embedding)
    Y_vec=np.array(Y_vec)
    length_sentence=np.array(length_sentence,dtype=int)
    print(X_embedding.shape,Y_vec.shape,length_sentence.shape)
    np.save(X_embedding_file,X_embedding)
    np.save(Y_vec_file,Y_vec)
    np.save(length_sentence_file,length_sentence)

    
if __name__=="__main__":
    main()
