import numpy as np 
import random
from biRNN_Attention import AttentionClassifier as Model
X_embedding_file = r'C:\Users\guan\Desktop\data\X_embedding_2and5_64.npy'
Y_vec_file = r'C:\Users\guan\Desktop\data\Y_vec_2and5_64.npy'
length_sentence_file=r'C:\Users\guan\Desktop\data\length_2and5_64.npy'
X_embedding_file_test = r'C:\Users\guan\Desktop\data\X_embedding_2and5_64.npy'
#this should be changed to a new data set when applying actually
#I split the data set for train and test instead.
Y_vec_file_test = r'C:\Users\guan\Desktop\data\Y_vec_2and5_64.npy'
length_sentence_file_test=r'C:\Users\guan\Desktop\data\length_2and5_64.npy'
n_models=7#the number of  models which would be ensembled
def main():
    #train()
    test()
def train():
    #the date set has been shuffled already,so split it directly
    inputs_resource=np.load(X_embedding_file)
    labels_resource=np.load(Y_vec_file)
    length_resource=np.load(length_sentence_file)
    n=int(len(length_resource)*0.8)
    model=Model()
    for i in range(n_models):
        index_bagging=[]
        for _ in range(n):
            index_bagging.append(random.randint(0,n-1))
        inputs=inputs_resource[index_bagging]
        labels=labels_resource[index_bagging]
        length=length_resource[index_bagging]
        model.fit(inputs,labels,length,'model{0}'.format(i))
def test():
    model=Model()
    inputs_resource=np.load(X_embedding_file_test)
    labels_resource=np.load(Y_vec_file_test)
    length_resource=np.load(length_sentence_file_test)
    index_split=int(len(length_resource)*0.8)
    inputs=inputs_resource[index_split:]
    labels=labels_resource[index_split:]
    length=length_resource[index_split:]
    prediction=np.zeros([len(length)],dtype=int)
    print(prediction.shape)
    for i in range(n_models):
        prediction+=model.pred_test(inputs,length,'model{0}'.format(i))
        print('prediction on model{0} finished'.format(i))
    n_split=int(n_models/2)+1
    prediction_final=np.where(prediction>=n_split,1,0)#prediction >= n_split means that there are at least n_split models predict the output as 1
    labels=np.argmax(labels,axis=-1)
    result=np.equal(prediction_final,labels)
    n_correct=np.sum(result)
    accuracy=float(n_correct)/float(len(result))
    print(accuracy)
if __name__ == '__main__':
    main()