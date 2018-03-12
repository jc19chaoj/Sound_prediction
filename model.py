import tensorflow as tf

class cnn_rnn:
""" AlexNet + RNN with LSTM model """

    def __init__(self, inputs, labels, fixed_layers=['conv1','conv2','conv3','conv4','conv5'], epoch=10, 
                 keep_prob=0.5, lstm_layers=2, num_class=0):
        self.epoch = epoch
        # inputs' and labels' data
        # self.inputs = []
        self.inputs = [tf.image.resize_image_with_crop_or_pad(frame,227,227) for frame in inputs]
        self.labels = labels
        # inputs' and labels' placeholder
        self.X = tf.placeholder(tf.float32, [None, 227, 227, 3])
        self.Y = tf.placeholder(tf.float32, [None, 42])
        
        self.num_class = num_class
        self.keep_prob = keep_prob
        # number of stacked LSTM layers
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
        self.fc6 = self.conv_layer('fc6', self.pool5, [5, 5], 4096)
        self.dropout6 = tf.nn.dropout(self.fc6, keep_prob=self.keep_prob, name='dropout6')
        # fc7
        self.fc7 = self.conv_layer('fc7', self.dropout6, [5, 5], 4096)
        self.dropout7 = tf.nn.dropout(self.fc7, keep_prob=self.keep_prob, name='dropout7')
        # fc8(output layer) -- not needed
        self.fc8 = self.conv_layer('fc8', self.dropout7, [1, 1], self.num_class, 
                                   output_layer=True)
        
        # for debugging
        if self.num_class>0:
            self.cnn_output = self.fc8
        else:
            self.cnn_output = self.fc7
        
        # LSTM layers
        with tf.variable_scope(name) as scope:
            w_init = tf.constant(np.random.rand(self.lstm_size, self.frequency_comp))
            b_init = tf.constant(np.zeros((1,self.frequency_comp)))
            weights = tf.get_variable('weights', initializer=w_init, dtype=tf.float32)
            biases = tf.get_variable('biases', initializer=b_init,  dtype=tf.float32)
                                 
            lstm = tf.contrib.rnn.LSTMCell(self.lstm_size)
            # dense = tf.layers.dense(lstm, 42)
            multi_layer_rnn = tf.contrib.rnn.MultiRNNCell([lstm] * self.num_layers)
            outputs, states = tf.nn.dynamic_rnn(multi_layer_rnn, self.inputs)

            logits = [tf.matmul(o, weights) + biases for o in outputs]
        
        # predictions = [tf.nn.softmax(l) for l in logits]

        loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(logit, label) 
                  for logit, label in zip(logits, labels)]
                                 
        mean_loss = tf.reduce_mean(loss)

        train_op = tf.train.AdamOptimizer(0.3).minimize(mean_loss)
        
    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # load pre-trained model
            self.load_weights()
            
            
            
            #for i in range(self.epoch):
                #sess.run()
    
    
    # Helper functions
    
    def conv_layer(self, name, inputs, filter_size, output_size, stride=1, padding='SAME', 
                   output_layer=False):
        # number of channels of input pictures
        input_channel = inputs.get_shape()[-1]
        
        # create 'weights' and 'biases' variables 
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[filter_size[0], filter_size[-1], input_channel, output_size])
            biases = tf.get_variable('biases', shape=[output_size])
            
            # convolute -> add bias -> activate
            conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding=padding)
            conv_bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
            activate = tf.nn.relu(conv_bias, name=scope.name)
            # don't return activation for output layer
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
        """ Pretrained model from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ """
        
        pretrained_weights = np.load('bvlc_alexnet.npy', encoding = 'bytes').item()

        for layer in pretrained_weights:
            if layer not in self.fixed_layer:
                with tf.variable_scope(layer, reuse = True):
                    # assign pretrained weights and variables to the trainable layers
                    for item in weights_dict[op_name]:
                        if len(item.shape) > 1:
                            weight_var = tf.get_variable('weights', trainable = False)
                            session.run(weight_var.assign(data))
                        else:
                            bias_var = tf.get_variable('biases', trainable = False)
                            session.run(bias_var.assign(data))
