import tensorflow as tf
import numpy as np
import os
X_embedding_file=r'C:\Users\guan\Desktop\data\X_embedding_2and5.npy'
Y_vec_file=r'C:\Users\guan\Desktop\data\Y_vec_2and5.npy'
class config(object):
    n_classes=2
    n_features=128
    n_sentence=50
    dropout_keep=1.0
    batch_size=512
    n_epoches=100
    n_epoches_adam=20
    lr=0.00001
    l2_rate= 10e-6
    filter_steps=[1,1,1]
    n_filters=200
    data_rate_train=0.8

class CNNClassifier(object):
    def __init__(self):
        self.config=config()
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss,self.accuracy = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op_adam(self.loss)
        self.saver = tf.train.Saver()
        self.save_path = os.path.abspath(os.path.join(os.path.curdir, 'models'))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def add_placeholders(self):
        self.input_placeholders=tf.placeholder(tf.float32,[None,self.config.n_sentence,self.config.n_features])
        self.label_placeholders=tf.placeholder(tf.float32,[None,self.config.n_classes])

    def create_feed_dict(self,input_batch,label_batch):
        feed_dict={self.input_placeholders:input_batch,self.label_placeholders:label_batch}
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
        self.W=tf.get_variable(name='W',shape=[n_units,self.config.n_classes],initializer = tf.contrib.layers.xavier_initializer())
        self.b=tf.Variable(tf.constant(0.0,shape=[self.config.n_classes]))
        prediction=tf.nn.xw_plus_b(conv_output_drop,self.W,self.b)
        # For convenience to use the tensorflow function,we return the values which can be transformed to y_hat only by a softmax function
        return prediction

    def add_loss_op(self,prediction):
        # Here we use softmax repeatedly,but the n_class is so small that the wasted time is too little.
        l2_loss=tf.nn.l2_loss(self.W)+tf.nn.l2_loss(self.b)
        loss=tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=self.label_placeholders)
        losses=tf.reduce_mean(loss)+self.config.l2_rate*l2_loss
        pred = tf.equal(tf.argmax(tf.nn.softmax(prediction),axis=1), tf.argmax(self.label_placeholders, axis=1))
        accuracy= tf.reduce_mean(tf.cast(pred,tf.float32))
        return losses,accuracy

    def add_training_op_adam(self,loss):
        adam_op=tf.train.AdamOptimizer()
        train_op=adam_op.minimize(loss)
        return train_op

    def add_training_op_SGD(self,loss):
        SGD_op=tf.train.GradientDescentOptimizer(self.config.lr)
        train_op=SGD_op.minimize(loss)
        return train_op

    def train_on_batch(self,sess,inputs_batch,labels_batch):
        feed=self.create_feed_dict(inputs_batch,labels_batch)
        _,loss,accuracy=sess.run([self.train_op,self.loss,self.accuracy],feed_dict=feed)
        return loss,accuracy

    def test(self,sess,inputs,labels):
        accuracy_list=[]
        data_size = len(labels)
        n_batches = int((data_size - 1) / self.config.batch_size) + 1
        for iter in range(n_batches):
            start=iter*self.config.batch_size
            end=min(start+self.config.batch_size,data_size)
            feed=self.create_feed_dict(inputs[start:end],labels[start:end])
            accuracy=sess.run([self.accuracy],feed_dict=feed)
            accuracy_list.append(accuracy)
        accuracy = np.mean(accuracy_list)
        return accuracy

    def run(self,inputs,labels,length):
        data_size=len(labels)
        assert data_size == len(inputs), 'inputs_size not equal to labels_size'
        #To keep the matching of inputs and lables ,we should use the shuffle index
        shuffle_index = np.random.permutation(np.arange(data_size))
        inputs_shuffle = inputs[shuffle_index]
        labels_shuffle = labels[shuffle_index]
        length_shuffle = length[shuffle_index]
        index_split=int(data_size*self.config.data_rate_train)
        inputs_train=inputs_shuffle[:index_split]
        labels_train=labels_shuffle[:index_split]
        length_train=length_shuffle[:index_split]
        inputs_test=inputs_shuffle[index_split:]
        labels_test=labels_shuffle[index_split:]
        length_test=length_shuffle[index_split:]
        print('inputs_train:',inputs_train.shape,'labels_train:',labels_train.shape)
        n_batches = int((index_split - 1) / self.config.batch_size)+1
        self.session=tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            print('Thare are {0} epoches'.format(self.config.n_epoches), 'each epoch has {0} steps'.format(n_batches))
            for iteration_epoch in range(self.config.n_epoches):
                print('epoch {0} =========================================================================='.format(iteration_epoch+1))
                train_loss=[]
                train_accuracy=[]
                for iteration_batch in range(n_batches):
                    start=iteration_batch*self.config.batch_size
                    end = min(start + self.config.batch_size, data_size)
                    loss,accuracy=self.train_on_batch(sess,inputs_train[start:end],labels_train[start:end])
                    iteration_batch+=1
                    if (iteration_batch % 50) == 0 :
                        print('training step:{0}'.format(iteration_batch))
                    train_loss.append(loss)
                    train_accuracy.append(accuracy)
                #The loss in train is the mean value of all loss on batches.
                train_loss_epoch=np.mean(train_loss)
                train_accuracy_epoch=np.mean(train_accuracy)
                print('loss_train:{:.3f}'.format(train_loss_epoch),'accuracy_train:{:.3f}'.format(train_accuracy_epoch))
                self.saver.save(sess,os.path.join(self.save_path,'model'),global_step=(iteration_epoch))
                accuracy_test=self.test(sess,inputs_test,labels_test)
                print('the accuracy in test :{:.3f}'.format(accuracy_test))

    def fit(self,inputs,labels):
        print('this a text classifier using CNN ')

        inputs=np.array(inputs)
        labels=np.array(labels)
        print('the shape of inputs:',inputs.shape)
        print('the shape of labels:',labels.shape)
        self.run(inputs,labels)
def main():
    model=CNNClassifier()
    inputs=np.load(X_embedding_file)
    labels=np.load(Y_vec_file)
    model.fit(inputs,labels)

if __name__ == '__main__':
    main()