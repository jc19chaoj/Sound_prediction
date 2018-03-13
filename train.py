import os
import numpy as np
import tensorflow as tf
from data import data_prepare
from model import cnn_rnn

def main_test():
    dp = data_prepare(train_size=10)
    dp.video_cap()
    dp.prediction_frames()
    dp.load_audio()

    input_data = dp.impact_frames
    label_data = dp.sound_features
    
    # test
    #input_data = [tf.zeros([300,300,3]) for i in range(30)]
    #label_data = [tf.zeros([42,]) for i in range(30)]
    print(len(label_data),len(label_data[0]),label_data[0])
    print(len(input_data),input_data[0].shape)
    print(input_data[5655].shape)
    
    # training configuration
    batch_size = 45
    epoch = 100
    input_data = input_data[0:225]
    input_data = [val for val in input_data for _ in range(3)] # up-sampling input data by 3
    label_data = label_data[0:675]
    #print(tf.convert_to_tensor(np.asarray(label_data)))
    
    model = cnn_rnn(batch_size=batch_size)

    model.train(input_data, label_data, epoch=epoch)
    
if __name__ == "__main__":
    main_test()
