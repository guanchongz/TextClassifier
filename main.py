import input
from keras.layers.core import Dense
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.callbacks import TensorBoard
import tensorflow as tf
X=input.gettotalX()
Y=input.gettotalY()
lenx=X.shape[1]
voca_size=input.vocal_size()
with tf.device('/gpu:0'):
    model=Sequential()
    model.add(Embedding(voca_size,lenx))

    model.add(GRU(60,dropout=0.1,return_sequences=True))
    model.add(GRU(30,dropout=0.1))
    model.add(Dense(3,activation="softmax"))
    tbCallBack = TensorBoard(log_dir='./Graph/sentiment_chinese', histogram_freq=0,
                            write_graph=True, write_images=True)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    model.fit(X,Y,validation_split=0.1,batch_size=512,epochs=30,verbose=2,callbacks=[tbCallBack])
    model.save('./model/model1.HDF5')

print("Saved model!")

