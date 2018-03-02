from text2vec import data
import tensorflow as tf
import numpy as np
import os
class config(object):
    n_classes=2
    n_features=64
    n_sentence=50
    dropout_keep=1.0
    batch_size=256
    n_epochs=200
    lr=0.0005
    l2_rate= 0.0
    filter_steps=[2,3,4,5]
    n_filters=100
    units_hidden=100
    learning_rate=0.0075
    data_rate_train=0.9

class CNNClassifier(object):
    def __init__(self):
        self.config=config()
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss,self.accuracy = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        self.saver = tf.train.Saver()
        self.save_path = os.path.abspath(os.path.join(os.path.curdir, 'models'))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def add_placeholders(self):
        self.input_placeholders=tf.placeholder(tf.float32,[None,self.config.n_sentence,self.config.n_features])
        self.label_placeholders=tf.placeholder(tf.float32,[None,self.config.n_classes])
        self.dropout_placeholder=tf.placeholder(tf.float32)

    def create_feed_dict(self,input_batch,label_batch,dropout=0.0):
        feed_dict={self.input_placeholders:input_batch,self.label_placeholders:label_batch,self.dropout_placeholder:dropout}
        return feed_dict

    def add_embeddings(self):
        #inputs are already sequences of word vectors
        return tf.expand_dims(self.input_placeholders,axis=-1)

    def add_prediction_op(self):
        x=self.add_embeddings()
        conv_output=[]
        for step in self.config.filter_steps :
            filter_shape=[step,self.config.n_features,1,self.config.n_filters]
            filter=tf.Variable(tf.truncated_normal(shape=filter_shape,stddev=0.1))
            bias=tf.Variable(tf.constant(0.0,shape=[self.config.n_filters]))
            # h_conv : shape =batch_szie * (n_sentence-step+1) * 1 * n_filters
            h_conv=tf.nn.conv2d(x,filter=filter,strides=[1,1,1,1],padding='VALID')
            h_relu=tf.nn.relu(tf.nn.bias_add(h_conv,bias))
            # h_pooling: shape = batch_size * 1 * 1 * n_filters
            h_pooling=tf.nn.max_pool(h_relu,ksize=[1,self.config.n_sentence-step+1,1,1],strides=[1,1,1,1],padding='VALID')
            conv_output.append(h_pooling)
        n_units=self.config.n_filters*len(self.config.filter_steps)
        conv_output=tf.concat(conv_output,axis=3)
        #the shape of input of DNN should be [batch,n_units]
        conv_output=tf.reshape(conv_output,shape=[-1,n_units])
        conv_output_drop=tf.nn.dropout(conv_output,self.config.dropout_keep)
        #single DNN layer
        self.W1=tf.get_variable(name='W',shape=[n_units,self.config.units_hidden],initializer = tf.contrib.layers.xavier_initializer())
        self.b1=tf.Variable(tf.constant(0.0,shape=[self.config.units_hidden]))
        x_hidden=tf.nn.relu(tf.nn.xw_plus_b(conv_output_drop,self.W1,self.b1))
        x_hidden_drop=tf.nn.dropout(x_hidden,self.config.dropout_keep)
        self.W2=tf.get_variable(name='W2',shape=[self.config.units_hidden,self.config.n_classes],initializer=tf.contrib.layers.xavier_initializer())
        self.b2=tf.Variable(tf.constant(0.0,shape=[self.config.n_classes]))
        self.prediction=tf.nn.xw_plus_b(x_hidden_drop,self.W2,self.b2)
        return self.prediction

    def add_loss_op(self,prediction):
        l2_loss=tf.nn.l2_loss(self.W1)+tf.nn.l2_loss(self.b1)+tf.nn.l2_loss(self.W2)+tf.nn.l2_loss(self.b2)
        loss=tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=self.label_placeholders)
        losses=tf.reduce_mean(loss)+self.config.l2_rate*l2_loss
        pred = tf.equal(tf.argmax(tf.nn.softmax(prediction),axis=1), tf.argmax(self.label_placeholders, axis=1))
        accuracy= tf.reduce_mean(tf.cast(pred,tf.float32))
        return losses,accuracy

    def add_training_op(self,loss):
        adam_op=tf.train.AdamOptimizer()
        train_op=adam_op.minimize(loss)
        return train_op


    def train_on_batch(self,sess,inputs_batch,labels_batch):
        feed=self.create_feed_dict(inputs_batch,labels_batch,self.config.dropout_keep)
        _,loss,accuracy=sess.run([self.train_op,self.loss,self.accuracy],feed_dict=feed)
        return loss,accuracy

    def creat_batches(self,inputs,labels):
        data_size=len(labels)
        n_batches=int((data_size-1)/self.config.batch_size)+1
        for epoch in range(self.config.n_epochs):
            for iter in range(n_batches):
                start=iter*self.config.batch_size
                end=min(start+self.config.batch_size,data_size)
                yield inputs[start:end],labels[start:end]

    def test(self,sess,inputs,labels):
        accuracy_list=[]
        for inputs_batch,labels_batch in self.creat_batches(inputs,labels):
            feed=self.create_feed_dict(inputs_batch,labels_batch,1.0)
            accuracy=sess.run([self.accuracy],feed_dict=feed)
            accuracy_list.append(accuracy)
        accuracy = np.mean(accuracy_list)
        return accuracy

    def run(self,inputs,labels):
        data_size=len(labels)
        assert data_size == len(inputs), 'inputs_size not equal to labels_size'
        shuffle_index = np.random.permutation(np.arange(data_size))
        inputs_shuffle = inputs[shuffle_index]
        labels_shuffle = labels[shuffle_index]
        index_split=int(data_size*self.config.data_rate_train)
        inputs_train=inputs_shuffle[:index_split]
        labels_train=labels_shuffle[:index_split]
        inputs_test=inputs_shuffle[index_split:]
        labels_test=labels_shuffle[index_split:]
        print('inputs_train:',inputs_train.shape,'labels_train:',labels_train.shape)
        n_batches = int((index_split - 1) / self.config.batch_size) + 1
        self.session=tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            iteration_batch=0
            iteration_epoch=1
            print('Thare are {0} epoches'.format(self.config.n_epochs),'each epoch has {0} steps'.format(n_batches))
            print('epoch 1 ==========================================================================')
            for inputs_batch,labels_batch in self.creat_batches(inputs_train,labels_train):
                loss,accuracy=self.train_on_batch(sess,inputs_batch,labels_batch)
                iteration_batch+=1
                if (iteration_batch % 20) == 0 :
                    print('training step:{0}'.format(iteration_batch),'training loss:{:.3f}'.format(loss),'training accuracy:{:.3f}'.format(accuracy))
                if iteration_batch == n_batches :
                    self.saver.save(sess,os.path.join(self.save_path,'model'),global_step=(iteration_epoch))
                    accuracy_test=self.test(sess,inputs_test,labels_test)
                    print('the accuracy in test :{:.3f}'.format(accuracy_test))
                    iteration_epoch+=1
                    iteration_batch=0
                    if iteration_epoch < n_batches*self.config.n_epochs :
                        print('epoch {0} =========================================================================='.format(iteration_epoch))

    def fit(self,inputs,labels):
        print('this a text classifier using CNN ')

        inputs=np.array(inputs)
        labels=np.array(labels)
        self.run(inputs,labels)
def main():
    model=CNNClassifier()
    inputs,labels=data()
    model.fit(inputs,labels)

if __name__ == '__main__':
    main()