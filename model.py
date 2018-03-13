import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from itertools import chain

class cnn_rnn:
    """ AlexNet + RNN with LSTM model """

    def __init__(self, 
                 fixed_layers=['conv1','conv2','conv3','conv4','conv5'],
                 keep_prob=0.5, 
                 batch_size=10, 
                 lstm_layers=3,
                 learn_rate=0.001, 
                 num_class=0):
        
        self.num_class = num_class
        self.keep_prob = keep_prob         
        self.learn_rate = learn_rate
                
        # minibatch config
        self.batch_size = batch_size
        
        # inputs' and labels' placeholder
        self.X = tf.placeholder(tf.float32, [self.batch_size, 340, 604, 3])
        self.Y = tf.placeholder(tf.float32, [self.batch_size, 42])
        
        # LSTM layers config
        self.num_layers = lstm_layers
        self.lstm_size = 256
        self.frequency_comp = 42
        
        # list of layers that is not going to be trained from scratch
        self.fixed_layers = fixed_layers

        # conv1
        self.conv1 = self.conv_layer('conv1', self.X, [11, 11], 96, 4, padding='VALID')
        print("conv1", self.conv1.get_shape())
        self.pool1 = self.pool_layer('pool1', self.conv1, [3, 3], 2)
        print("pool1", self.pool1.get_shape())
        # conv2
        self.conv2 = self.conv_layer('conv2', self.pool1, [5, 5], 256, group=2)
        self.pool2 = self.pool_layer('pool2', self.conv2, [3, 3], 2)
        # conv3
        self.conv3 = self.conv_layer('conv3', self.pool2, [3, 3], 384)
        # conv4
        self.conv4 = self.conv_layer('conv4', self.conv3, [3, 3], 384, group=2)
        # conv5
        self.conv5 = self.conv_layer('conv5', self.conv4, [3, 3], 256, group=2)
        self.pool5 = self.pool_layer('pool5', self.conv5, [3, 3], 2)
        # fc6
        self.fc6 = self.conv_layer('fc6', self.pool5, [], 4096, padding="VALID", fc_layer=True)
        self.dropout6 = tf.nn.dropout(self.fc6, keep_prob=self.keep_prob, name='dropout6')
        # fc7
        self.fc7 = self.conv_layer('fc7', self.dropout6, [], 4096, padding="VALID", fc_layer=True)
        self.dropout7 = tf.nn.dropout(self.fc7, keep_prob=self.keep_prob, name='dropout7')
        # fc8 (classification layer)
        self.fc8 = self.conv_layer('fc8', self.dropout7, [], self.num_class, 
                                   output_layer=True, padding="VALID", fc_layer=True)
        
        # feature map or classifier output
        if self.num_class>0:
            self.cnn_output = self.fc8
        else:
            self.cnn_output = self.fc7
        
        # reshaped to [batch_size, time_step, LSTM_size] for feeding to LSTM
        self.cnn_output = tf.reshape(self.cnn_output, shape=[self.batch_size,1,4096])
        
        # RNN layers
        with tf.variable_scope('rnn'):
            w_init = tf.constant(np.random.rand(self.lstm_size, self.frequency_comp).astype(np.float32))
            b_init = tf.constant(np.zeros((1,self.frequency_comp)).astype(np.float32))
            weights = tf.get_variable('weights', initializer=w_init, dtype=tf.float32)
            biases = tf.get_variable('biases', initializer=b_init,  dtype=tf.float32)
            
            # lstm layers   
            lstm = lambda lstm_size: tf.contrib.rnn.LSTMCell(lstm_size)
            multi_layer_rnn = tf.nn.rnn_cell.MultiRNNCell([lstm(self.lstm_size) for _ in range(self.num_layers)])
            outputs, states = tf.nn.dynamic_rnn(multi_layer_rnn, self.cnn_output, dtype=tf.float32)

            # dense layer
            outputs = tf.reshape(outputs, shape=[self.batch_size,256])
            self.logits = tf.matmul(outputs, weights) + biases
            
            
        # define loss function
        rho = lambda r: tf.log(tf.add(tf.square(r),1/625))
        loss = rho(tf.nn.l2_loss(self.Y-self.logits))                        
        self.mean_loss = tf.reduce_mean(loss)
        self.total_loss = tf.reduce_sum(loss)
        
        # choose optimizer
        #self.optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.mean_loss)
        self.optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)
        
        #return self.logits
        
    def train(self, 
             inputs, 
             labels,
             test_inputs,
             crop_inputs=False,
             epoch=10):
        
        # number of training epoch
        self.epoch = epoch
        
        # inputs' and labels' data
        # self.inputs = inputs
        self.inputs = [frame/255 for frame in inputs]
        # self.inputs = [tf.image.convert_image_dtype(frame, dtype=tf.float32) for frame in inputs]
        if crop_inputs:
            self.inputs = [tf.image.resize_image_with_crop_or_pad(self.inputs,227,227) for frame in inputs]
        self.labels = labels
        
        print(len(self.inputs),self.inputs[0])
        print(len(self.labels),self.labels[0].shape)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # load pre-trained model
            self.load_weights(sess)
            
            # training
            print("Start training...")
            for i in range(self.epoch):
                print("epoch: ", i)
                self.batch_index = 0
                self.shuffle_data()
                for j in range(len(self.inputs)//self.batch_size):
                    inputs_batch, labels_batch = self.next_batch()
                    #print(inputs_batch.shape, labels_batch.shape)
                    
                    loss, _ = sess.run([self.total_loss, self.optimizer], feed_dict={self.X: inputs_batch, self.Y: labels_batch})
                print("    error: ", loss)
            
            self.test(sess, test_inputs)


    def test(self, sess, test_inputs):
        """evaluate prediction"""
        predict_sound_feature = []
        for w in range(len(test_inputs)//self.batch_size):        
            logits = sess.run(self.logits, feed_dict={self.X: test_inputs})
            predict_sound_feature.extend(logits)
            
        predict_sound_feature = np.asarray(predict_sound_feature)
        plt.figure()
        plt.imshow(np.asarray(predict_sound_feature.T))
        plt.colorbar(orientation='vertical')
        plt.show()
    
    # Helper functions
    def conv_layer(self, name, inputs, filter_size, output_size, stride=1, padding='SAME', group=1, 
                   output_layer=False, fc_layer=False):
        # number of channels of input pictures
        input_shape = inputs.get_shape()
        input_height, input_width, input_channel = input_shape[1:]
        
        # create 'weights' and 'biases' variables 
        with tf.variable_scope(name) as var_scope:
            if fc_layer:
                weights = tf.get_variable("weights", shape=[input_height, input_width, input_channel, output_size])
            else:
                weights = tf.get_variable('weights', shape=[filter_size[0], filter_size[-1], input_channel, output_size])
            biases = tf.get_variable('biases', shape=[output_size])
            
            conv_op = lambda x, w: tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
            
            if group>1:
                inputs_split = tf.split(inputs, group, 3)
                weights_split = tf.split(weights, group, 2)
                outputs_split = [conv_op(x,w) for x,w in zip(inputs_split, weights_split)]
                
                conv = tf.concat(outputs_split, 2)
            else:
                conv = conv_op(inputs, weights)
                
            conv_bias = tf.nn.bias_add(conv, biases)
            activate = tf.nn.relu(conv_bias, name=var_scope.name)
            
            if output_layer:
                return conv_bias
            else:
                return activate 
        
    def pool_layer(self, name, inputs, filter_size, stride=1):
        return tf.nn.max_pool(inputs, 
                              [1, filter_size[0], filter_size[-1], 1], 
                              [1, stride, stride, 1], 
                              padding='VALID', 
                              name=name)
    
    def load_weights(self, session):
        """Assign pretrained weights and variables to the trainable layers
           Pretrained model downloaded from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/"""
        pretrained_weights = np.load('bvlc_alexnet.npy', encoding = 'bytes').item()

        for layer in pretrained_weights:
            if layer in self.fixed_layers:
                with tf.variable_scope(layer, reuse = True):
                    for item in pretrained_weights[layer]:
                        if len(item.shape) > 1:
                            weight_var = tf.get_variable('weights', trainable = False)
                            session.run(weight_var.assign(item))
                        else:
                            bias_var = tf.get_variable('biases', trainable = False)
                            session.run(bias_var.assign(item))
                            
    def next_batch(self):
        """load next minibatch of inputs and labels"""
        if self.batch_index <= len(self.inputs)-self.batch_size:
            
            
            input_batch = self.inputs[self.batch_index:self.batch_index+self.batch_size]
            input_batch = np.asarray(input_batch).astype(np.float32)
            
            label_batch = self.labels[self.batch_index:self.batch_index+self.batch_size]
            label_batch = np.asarray(label_batch).astype(np.float32)
            
            self.batch_index += self.batch_size
            
        return input_batch, label_batch
        
    def shuffle_data(self):
        impact_video_shuffled = [self.inputs[i:i+15] for i in range(len(self.inputs)//15)]
        impact_audio_shuffled = [self.labels[i:i+45] for i in range(len(self.labels)//45)]
        
        shuffle_together = list(zip(impact_video_shuffled, impact_audio_shuffled))
        shuffle(shuffle_together)
        
        impact_video_shuffled = [item for item,_ in shuffle_together]
        impact_audio_shuffled = [item for _,item in shuffle_together]
        
        self.input_data = list(chain.from_iterable(impact_video_shuffled))
        self.label_data = list(chain.from_iterable(impact_audio_shuffled))
        
