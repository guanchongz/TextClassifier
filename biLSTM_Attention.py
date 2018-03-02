from text2vec import data
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import rnn
class config(object):
    n_classes=2
    n_features=64
    n_layers=2
    dropout_keep=0.6
    batch_size=256
    n_time_steps=50
    l2_loss_rate= 10e-7
    n_epoches=20
    data_rate_train=0.9

class AttentionClassifier(object):
    def __init__(self):
        self.config = config()
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss, self.accuracy = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        self.saver = tf.train.Saver()
        self.save_path = os.path.abspath(os.path.join(os.path.curdir, 'models'))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def add_placeholders(self):
        self.input_placeholders=tf.placeholder(tf.float32,[None,self.config.n_time_steps,self.config.n_features])
        self.label_placeholders=tf.placeholder(tf.float32,[None,self.config.n_classes])
        self.dropout_placeholder=tf.placeholder(tf.float32)

    def create_feed_dict(self,input_batch,label_batch,dropout=0.0):
        feed_dict={self.input_placeholders:input_batch,self.label_placeholders:label_batch,self.dropout_placeholder:dropout}
        return feed_dict

    def attention(self,inputs):
        #inputs.shape:[batch_size,n_time_steps,n_features*2]
        attention_size=self.config.n_features*2
        self.W_attention=tf.Variable(tf.truncated_normal([attention_size,attention_size],stddev=0.1),dtype=tf.float32)
        self.b_attention=tf.Variable(tf.constant(0.0,shape=[attention_size]))
        self.u_attention=tf.Variable(tf.truncated_normal([attention_size],stddev=0.1))
        #same caclulate on every time state of every sentence on batch,so reshape inputs
        h=tf.reshape(inputs,[-1,attention_size])
        #h.shape:[batch_size*n_time_step,n_features*2
        hW=tf.tanh(tf.matmul(h,self.W_attention)+tf.reshape(self.b_attention,[1,-1]))
        #hW.shape:[batch_size*n_time_step,n_features*2
        hWu=tf.matmul(hW,tf.reshape(self.u_attention,[-1,1]))
        #hWu.shape:[batch_size*n_time_step,1]
        importance=tf.reshape(hWu,[-1,self.config.n_time_steps])
        #the wight for the hidden_state vector in every time step for every sentence in  batch
        wight=tf.nn.softmax(importance)
        wight=tf.expand_dims(wight,axis=2)
        output=tf.reduce_sum(inputs*wight,1)
        #output.shape:[batch_size,n_features*2]
        return output

    def add_prediction_op(self):
        x=self.input_placeholders
        def lstm_cell():
            lstm = rnn.BasicLSTMCell(self.config.n_features)
            drop = rnn.DropoutWrapper(lstm,output_keep_prob=self.config.dropout_keep)
            return drop
        cell_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(self.config.n_layers)])
        cell_bw = rnn.MultiRNNCell([lstm_cell() for _ in range(self.config.n_layers)])
        (rnn_output,_)=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,x,dtype=tf.float32)
        rnn_output=tf.concat(rnn_output,-1)
        h=self.attention(rnn_output)
        h_drop=tf.nn.dropout(h,self.config.dropout_keep)
        self.W=tf.Variable(tf.truncated_normal([self.config.n_features*2,self.config.n_classes],stddev=0.1))
        self.b=tf.Variable(tf.constant(0.0,shape=[self.config.n_classes]))
        y_hat = tf.nn.xw_plus_b(h_drop, self.W, self.b)
        return y_hat

    def add_loss_op(self,y_hat):
        l2_loss=tf.nn.l2_loss(self.W)+tf.nn.l2_loss(self.b)+tf.nn.l2_loss(self.W_attention)+ \
                        tf.nn.l2_loss(self.b_attention)+tf.nn.l2_loss(self.u_attention)
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat,labels=self.label_placeholders))
        loss+=self.config.l2_loss_rate*l2_loss
        prediction=tf.nn.softmax(y_hat)
        pred=tf.equal(tf.argmax(prediction,axis=1),tf.argmax(self.label_placeholders,axis=1))
        accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
        return loss,accuracy

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
        for epoch in range(self.config.n_epoches):
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

    def run(self, inputs, labels):
        data_size = len(labels)
        assert data_size == len(inputs), 'inputs_size not equal to labels_size'
        shuffle_index = np.random.permutation(np.arange(data_size))
        inputs_shuffle = inputs[shuffle_index]
        labels_shuffle = labels[shuffle_index]
        index_split = int(data_size * self.config.data_rate_train)
        inputs_train = inputs_shuffle[:index_split]
        labels_train = labels_shuffle[:index_split]
        inputs_test = inputs_shuffle[index_split:]
        labels_test = labels_shuffle[index_split:]
        print('inputs_train:', inputs_train.shape, 'labels_train:', labels_train.shape)
        n_batches = int((index_split - 1) / self.config.batch_size) + 1
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            iteration_batch = 0
            iteration_epoch = 1
            print('Thare are {0} epoches'.format(self.config.n_epoches),
                  'each epoch has {0} steps'.format(n_batches))
            print('epoch 1 ==========================================================================')
            for inputs_batch, labels_batch in self.creat_batches(inputs_train, labels_train):
                loss, accuracy = self.train_on_batch(sess, inputs_batch, labels_batch)
                iteration_batch += 1
                if (iteration_batch % 20) == 0:
                    print('training step:{0}'.format(iteration_batch), 'training loss:{:.3f}'.format(loss),'training accuracy:{:.3f}'.format(accuracy))
                if iteration_batch == n_batches:
                    self.saver.save(sess, os.path.join(self.save_path, 'model'), global_step=(iteration_epoch))
                    accuracy_test = self.test(sess, inputs_test, labels_test)
                    print('the accuracy in test :{:.3f}'.format(accuracy_test))
                    iteration_epoch += 1
                    iteration_batch = 0
                    if iteration_epoch < n_batches * self.config.n_epoches:
                        print('epoch {0} =========================================================================='.format(iteration_epoch))

    def fit(self, inputs, labels):
        print('this a text classifier using CNN ')
        inputs = np.array(inputs)
        labels = np.array(labels)
        self.run(inputs, labels)

def main():
    model = AttentionClassifier()
    inputs, labels = data()
    model.fit(inputs, labels)

if __name__ == '__main__':
    main()