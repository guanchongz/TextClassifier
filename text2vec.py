import codecs
from queue import Queue
import numpy as np
from gensim.models import Word2Vec
def main():
    dimention=32#the dimention of word vector
    n_class=2#the number of classes in label
    len_max=50#the maximum length in sentences

    Xfile=r'C:\Users\guan\Desktop\data\X_2and5.txt'
    Yfile=r'C:\Users\guan\Desktop\data\Y_2and5.txt'
    X_embedding_file=r'C:\Users\guan\Desktop\data\X_embedding_2and5_32'
    Y_vec_file=r'C:\Users\guan\Desktop\data\Y_vec_2and5_32'
    length_sentence_file=r'C:\Users\guan\Desktop\data\length_2and5_32'
    word2vec=r'C:\Users\guan\Desktop\data\word2vec_32.model'
    Xf=codecs.open(Xfile,'r',encoding='UTF-8')
    Yf=codecs.open(Yfile,'r',encoding='UTF-8')
    X=[]
    Y=Queue()
    #Abandon all blanks when load the file
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
    vector_blank=np.array([0.0]*dimention)#When the length of sentences is less than the maximum length ,fill it by 0.0 vectors
    for sentence in X:
        sentence_embedding=[]
        for word in sentence :
            try:
                sentence_embedding.append(word_vectors.wv[word])#transform the words to vectors
            except:
                continue
        #get the labels which mate the sentence
        index=Y.get()
        #Fill the sentences to miximum length by 0 vectors
        length=len(sentence_embedding)
        if length>0 and length<=len_max :
            for i in range(len_max-length):
                sentence_embedding.append(vector_blank)
            X_embedding.append(np.array(sentence_embedding))
            #transform the int numbers to one-hot vectors
            label_vec=[0]*n_class
            label_vec[index]=1
            #only when the sentence has been appended,append the labels
            length_sentence.append(length)
            Y_vec.append(np.array(label_vec,dtype=int))
    X_embedding=np.array(X_embedding)
    Y_vec=np.array(Y_vec)
    length_sentence=np.array(length_sentence,dtype=int)
    shuffle_index = np.random.permutation(len(length_sentence))
    inputs_shuffle = X_embedding[shuffle_index]
    labels_shuffle = Y_vec[shuffle_index]
    length_shuffle = length_sentence[shuffle_index]
    #Observe the shape
    print(X_embedding.shape,Y_vec.shape,length_sentence.shape)
    #save numpy arrays
    np.save(X_embedding_file,inputs_shuffle)
    np.save(Y_vec_file,labels_shuffle)
    np.save(length_sentence_file,length_shuffle)

    
if __name__=="__main__":
    main()
