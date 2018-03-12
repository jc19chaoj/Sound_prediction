import os
import numpy as np
import tensorflow as tf
from data import data_prepare
from model import cnn_rnn

def main_test():
    dp = data_prepare(train_size=10)
    dp.video_cap()
    dp.prediction_frames()
    dp.load_audio(1)

    input_data = dp.impact_frames
    label_data = dp.sound_features
    print(len(label_data),len(label_data[0]),label_data[0])
    print(len(input_data))
    print(input_data[5655].shape)
    
    input_data = input_data[0:10]
    input_data = [val for val in input_data for _ in range(3)]
    print(len(input_data))
    
    label_data = label_data[0:30]
    model = cnn_rnn(input_data, label_data)

if __name__ == "__main__":
    main_test()
