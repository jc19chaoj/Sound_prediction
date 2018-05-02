# Chao Jiang
# Model: CNN (AlexNet) + RNN w/ LSTM
import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from itertools import chain
from send_email import *
from config import *

class cnn_rnn:
    """ AlexNet + RNN with LSTM model """
    def __init__(self, 
                 fixed_layers=['conv1','conv2','conv3','conv4','conv5'],
                 keep_prob=1, 
                 batch_size=45, 
                 lstm_layers=2,
                 learn_rate=0.001, 
                 num_class=0):
        
        self.num_class = num_class
        self.keep_prob = keep_prob         
        self.learn_rate = learn_rate
                
        # minibatch config
        self.batch_size = batch_size
        
        # inputs' and labels' placeholder
        self.X = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.Y = tf.placeholder(tf.float32, [None, 45, 42])
        
        # LSTM layers config
        self.num_layers = lstm_layers
        self.lstm_size = 256
        self.frequency_comp = 42
        
        # list of layers that is not going to be trained from scratch
        self.fixed_layers = fixed_layers

        # conv1
        self.conv1 = self.conv_layer('conv1', self.X, [11, 11], 96, 4, padding='VALID')
        #print("conv1", self.conv1.get_shape())
        self.pool1 = self.pool_layer('pool1', self.conv1, [3, 3], 2)
        #print("pool1", self.pool1.get_shape())
        # conv2
        self.conv2 = self.conv_layer('conv2', self.pool1, [5, 5], 256, group=2)
        #print("conv2", self.conv2.get_shape())
        self.pool2 = self.pool_layer('pool2', self.conv2, [3, 3], 2)
        #print("pool2", self.pool2.get_shape())
        # conv3
        self.conv3 = self.conv_layer('conv3', self.pool2, [3, 3], 384)
        #print("conv3", self.conv3.get_shape())
        # conv4
        self.conv4 = self.conv_layer('conv4', self.conv3, [3, 3], 384, group=2)
        #print("conv4", self.conv4.get_shape())
        # conv5
        self.conv5 = self.conv_layer('conv5', self.conv4, [3, 3], 256, group=2)
        self.pool5 = self.pool_layer('pool5', self.conv5, [3, 3], 2)
        # Flatten output from conv5
        self.reshaped_pool5 = self.flatten_tensor(self.pool5)
        # fc6
        #self.fc6 = self.conv_layer('fc6', self.pool5, [], 4096, padding="VALID", fc_layer=True)
        self.fc6 = self.fc_layer('fc6', self.reshaped_pool5, 4096, trainable=True)
        self.dropout6 = tf.nn.dropout(self.fc6, keep_prob=self.keep_prob, name='dropout6')
        # fc7
        #self.fc7 = self.conv_layer('fc7', self.dropout6, [], 4096, padding="VALID", fc_layer=True)
        self.fc7 = self.fc_layer('fc7', self.dropout6, 4096, trainable=True)
        self.dropout7 = tf.nn.dropout(self.fc7, keep_prob=self.keep_prob, name='dropout7')

        #self.cnn_output = self.fc7

        # use min max normalization
        self.cnn_output = tf.div(tf.subtract(self.fc7, tf.reduce_min(self.fc7)), tf.subtract(tf.reduce_max(self.fc7), tf.reduce_min(self.fc7)))

        # reshaped to [batch_size, time_step, LSTM_size] for feeding to LSTM
        self.cnn_output = tf.reshape(self.cnn_output, shape=[-1,45,4096])

        # RNN layers
        with tf.variable_scope('rnn'):
            # lstm layers   
            lstm = tf.contrib.rnn.BasicLSTMCell
            cells = []
            for _ in range(self.num_layers):
                cells.append(lstm(self.lstm_size, forget_bias=1.0))

            multi_layer_rnn = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            outputs, states = tf.nn.dynamic_rnn(multi_layer_rnn, self.cnn_output, dtype=tf.float32)
            #outputs, states = tf.contrib.rnn.static_rnn(multi_layer_rnn, self.cnn_output, dtype=tf.float32, sequence_length=45)

            print("RNN output",outputs.get_shape())
            self.rnn_output = outputs
            
            # dense layer
            w_init = tf.constant(np.random.rand(self.lstm_size, self.frequency_comp).astype(np.float32))
            b_init = tf.constant(np.zeros((1,self.frequency_comp)).astype(np.float32))
            weights = tf.get_variable('weights', initializer=w_init, dtype=tf.float32)
            biases = tf.get_variable('biases', initializer=b_init,  dtype=tf.float32)

            outputs = tf.reshape(outputs, shape=[-1,256])
            self.logits = tf.matmul(outputs, weights) + biases
            self.logits = tf.reshape(self.logits, shape=[-1, 45, 42])

            
            
        # define loss function
        rho = lambda r: tf.log(tf.add(tf.square(r), 1/625))
        self.loss = rho(tf.nn.l2_loss(tf.add(self.Y, -self.logits))) 
                      
        # choose optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learn_rate, 0.8).minimize(self.loss)
        #self.optimizer = tf.train.RMSPropOptimizer(self.learn_rate).minimize(self.loss)
        #self.optimizer = tf.train.MomentumOptimizer(self.learn_rate,0.9).minimize(self.loss)
        #self.optimizer = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

        
    def train(self, 
             sess, 
             if_restore,
             inputs, 
             labels,
             epoch,
             check_point=100):

        self.epoch = epoch
        self.inputs = inputs
        self.labels = labels
        
        print(len(self.inputs),self.inputs[0].shape)
        print(len(self.labels),self.labels[0].shape)

        sess.run(tf.global_variables_initializer())

        # load pre-trained AlexNet model
        self.load_weights(sess, trainable=False)
        
        # training
        if if_restore:
            restore_meta = tf.train.import_meta_graph(os.path.join(LOGDIR, 'model.ckpt-{}.meta'.format(check_point)))
            restore_meta.restore(sess, tf.train.latest_checkpoint(LOGDIR))
            print("Resume training...")
        else:
            print("Start training...")

        for i in range(self.epoch):
            self.batch_index = 0
            self.shuffle_data()
            avg_loss = 0
            for j in range(len(self.inputs)//self.batch_size):
                inputs_batch, labels_batch = self.next_batch()
                loss,_ = sess.run([self.loss, self.optimizer], feed_dict={self.X: inputs_batch, self.Y: labels_batch})
                avg_loss += loss/(len(self.inputs)//self.batch_size) 
            
            print("epoch: ", i)
            print("    error: ", avg_loss)

            if i==10 or i==50 or i==100 or i==200 :
                self.test(sess, inputs_batch[0:45])
                #self.shuffle_data()
            if i%50 == 0:
                self.saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)


    def test(self, sess, test_inputs):
        """evaluate training"""
        # sess.run(tf.global_variables_initializer())
        # self.load_weights(sess, trainable=False)

        predict_sound_feature = []
        for w in range(len(test_inputs)//45):        
            logits = sess.run(self.logits, feed_dict={self.X: test_inputs[w*self.batch_size:(w+1)*self.batch_size]})
            predict_sound_feature.append(logits.reshape(45,42))
            
        for prediction in predict_sound_feature:
            plt.figure()
            plt.imshow(np.asarray(prediction.T))
            plt.colorbar(orientation='vertical')
            plt.show()
            plt.savefig('test_result.png')


    def inference(self, sess, frame, check_point):
        """predict sound from new video inputs"""
        restore_meta = tf.train.import_meta_graph(os.path.join(LOGDIR, 'model.ckpt-{}.meta'.format(check_point)))
        restore_meta.restore(sess, tf.train.latest_checkpoint(LOGDIR))
        print("Start inference...")
        cnn_output, rnn_output, predict = sess.run([self.cnn_output, self.rnn_output, self.logits], feed_dict={self.X: frame})
        print("Prediction shape:", predict.shape)
        print("CNN OUT:", cnn_output)
        print("RNN OUT:", rnn_output)
        return predict

    ######################################################################
    # Helper functions:

    def flatten_tensor(self, inputs):
        input_height, input_width, input_channel = inputs.get_shape()[1:]
        input_units = input_height*input_width*input_channel
        reshaped_inputs = tf.reshape(inputs, shape=[-1, 1, input_units])
        return reshaped_inputs

    def conv_layer(self, name, inputs, filter_size, output_size, stride=1, padding='SAME', group=1, 
                   classifier_layer=False):
        # number of channels of input pictures
        input_shape = inputs.get_shape()
        input_height, input_width, input_channel = input_shape[1:]
        
        # create 'weights' and 'biases' variables 
        with tf.variable_scope(name) as var_scope:
            if group > 1:
                weights = tf.get_variable('weights', shape=[filter_size[0], filter_size[-1], int(input_channel)/group, output_size])
            else:
                weights = tf.get_variable('weights', shape=[filter_size[0], filter_size[-1], input_channel, output_size])
            biases = tf.get_variable('biases', shape=[output_size])
            
            conv_op = lambda x, w: tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
            
            if group>1:
                inputs_split = tf.split(inputs, group, 3)
                weights_split = tf.split(weights, group, 3)
                
                outputs_split = [conv_op(x,w) for x,w in zip(inputs_split, weights_split)]
                #print("split", inputs_split, weights_split, outputs_split)
                conv = tf.concat(outputs_split, 3)
            else:
                conv = conv_op(inputs, weights)
                
            conv_bias = tf.nn.bias_add(conv, biases)
            conv_bias = tf.reshape(conv_bias, [-1] + conv.get_shape().as_list()[1:])
            activate = tf.nn.relu(conv_bias, name=var_scope.name)
            
            return activate 
        
    def pool_layer(self, name, inputs, filter_size, stride=1):
        return tf.layers.max_pooling2d(inputs, 
                              [filter_size[0], filter_size[-1]], 
                              [stride, stride], 
                              padding='valid', 
                              name=name)
    
    def fc_layer(self, name, inputs, output_units, trainable, activation=tf.nn.relu):
        input_units = inputs.get_shape()[-1]

        with tf.variable_scope(name) as var_scope:
            w_init = tf.constant_initializer(np.random.rand(input_units, output_units).astype(np.float32))
            b_init = tf.constant_initializer(np.zeros((1, output_units)).astype(np.float32))

            return tf.layers.dense(inputs, output_units, activation=tf.nn.relu, 
                                name=name, trainable=trainable,
                                kernel_initializer=w_init,
                                bias_initializer=b_init)
    
    def load_weights(self, session, trainable=False):
        """Assign pretrained weights and variables to the trainable layers
           Pretrained model downloaded from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/"""
        pretrained_weights = np.load('bvlc_alexnet.npy', encoding = 'bytes').item()

        for layer in pretrained_weights:
            if layer in self.fixed_layers:
                with tf.variable_scope(layer, reuse = True):
                    for item in pretrained_weights[layer]:
                        if len(item.shape) > 1:
                            weight_var = tf.get_variable('weights', trainable = trainable)
                            session.run(weight_var.assign(item))
                        else:
                            bias_var = tf.get_variable('biases', trainable = trainable)
                            session.run(bias_var.assign(item))
                            
    def next_batch(self):
        """load next minibatch of inputs and labels"""
        if self.batch_index <= len(self.input_data)-self.batch_size:
            
            input_batch = self.input_data[self.batch_index:self.batch_index+self.batch_size]
            input_batch = np.asarray(input_batch).astype(np.float32)
            
            label_batch = self.label_data[self.batch_index:self.batch_index+self.batch_size]
            label_batch = np.asarray(label_batch).astype(np.float32)
            
            self.batch_index += self.batch_size
            
        return input_batch, label_batch.reshape(-1, 45, 42)
        
    
    def shuffle_data(self):
        impact_video_shuffled = [self.inputs[i:i+45] for i in range(len(self.inputs)//45)]
        impact_audio_shuffled = [self.labels[i:i+45] for i in range(len(self.labels)//45)]
        #print(len(impact_video_shuffled), len(impact_audio_shuffled))

        shuffle_together = list(zip(impact_video_shuffled, impact_audio_shuffled))
        shuffle(shuffle_together)
        
        self.input_data = []
        self.label_data = []
        for item1, item2 in shuffle_together:
            self.input_data.extend(item1)
            self.label_data.extend(item2)
