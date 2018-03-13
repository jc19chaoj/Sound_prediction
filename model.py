import tensorflow as tf
import numpy as np

class cnn_rnn:
    """ AlexNet + RNN with LSTM model """

    def __init__(self, 
                 fixed_layers=['conv1','conv2','conv3','conv4','conv5'],
                 keep_prob=0.5, 
                 batch_size=10, 
                 lstm_layers=2, 
                 num_class=0):
        
        self.num_class = num_class
        self.keep_prob = keep_prob         
                
        # minibatch config
        self.batch_size = batch_size
        self.batch_index = 0
        
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
        self.pool1 = self.pool_layer('pool1', self.conv1, [3, 3], 2)
        # conv2
        self.conv2 = self.conv_layer('conv2', self.pool1, [5, 5], 256)
        self.pool2 = self.pool_layer('pool2', self.conv2, [3, 3], 2)
        # conv3
        self.conv3 = self.conv_layer('conv3', self.pool2, [3, 3], 384)
        # conv4
        self.conv4 = self.conv_layer('conv4', self.conv3, [3, 3], 384)
        # conv5
        self.conv5 = self.conv_layer('conv5', self.conv4, [3, 3], 256)
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
            logits = tf.matmul(outputs, weights) + biases
            
            
        # define loss function
        rho = lambda r: tf.log(tf.add(tf.square(r),1/625))
        loss = rho(tf.nn.l2_loss(self.Y-logits))                        
        self.mean_loss = tf.reduce_mean(loss)
        
        # choose optimizer
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.mean_loss)
        
    def train(self, 
             inputs, 
             labels,
             crop_inputs=False,
             epoch=10):
        
        # number of training epoch
        self.epoch = epoch
        
        # inputs' and labels' data
        self.inputs = inputs
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
            
            print("Start training...")
            
            inputs_batch, labels_batch = self.next_batch()
            
            for i in range(self.epoch):
                print("epoch: ", i)
                
                for j in range(len(self.inputs)//self.batch_size):
                    loss, _ = sess.run([self.mean_loss, self.optimizer], feed_dict={self.X: inputs_batch, self.Y: labels_batch})
             
                print("error: ", loss)
                    
    
    # Helper functions
    def conv_layer(self, name, inputs, filter_size, output_size, stride=1, padding='SAME', 
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
            
            conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding=padding)
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
                              padding='SAME', 
                              name=name)
    
    def load_weights(self, session):
        """ 
        Assign pretrained weights and variables to the trainable layers
        Pretrained model downloaded from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ 
        """
        pretrained_weights = np.load('bvlc_alexnet.npy', encoding = 'bytes').item()

        for layer in pretrained_weights:
            if layer not in self.fixed_layers:
                with tf.variable_scope(layer, reuse = True):
                    for item in pretrained_weights[layer]:
                        if len(item.shape) > 1:
                            weight_var = tf.get_variable('weights', trainable = False)
                            session.run(weight_var.assign(weight_var))
                        else:
                            bias_var = tf.get_variable('biases', trainable = False)
                            session.run(bias_var.assign(bias_var))
                            
    def next_batch(self):
        """load next minibatch of inputs and labels"""
        if self.batch_index <= len(self.inputs)-self.batch_size:
        
            inputs_batch = self.inputs[self.batch_index:self.batch_index+self.batch_size]
            inputs_batch = np.asarray(inputs_batch).astype(np.float32)
            
            labels_batch = self.labels[self.batch_index:self.batch_index+self.batch_size]
            labels_batch = np.asarray(labels_batch).astype(np.float32)
            
            self.batch_index += self.batch_size
            
        return inputs_batch, labels_batch
        

        
