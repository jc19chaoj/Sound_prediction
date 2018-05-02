import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import *
from model_alexnet import cnn_rnn
from random import shuffle
from itertools import chain

from config import *

def main(if_shuffle=False, if_restore=False, check_point=100):
    dp = data_prepare(train_size=[0,100], impact_size=5)
    dp.video_cap()
    dp.prediction_frames()
    dp.load_audio()

    input_data_raw = dp.impact_frames
    label_data = dp.sound_features

    if if_shuffle:
        impact_video_shuffled = [input_data_raw[i:i+15] for i in range(len(input_data_raw)//15)]
        impact_audio_shuffled = label_data
        print(len(impact_video_shuffled), len(impact_audio_shuffled))
            
        shuffle_together = list(zip(impact_video_shuffled, impact_audio_shuffled))
        shuffle(shuffle_together)
        
        impact_video_shuffled = []
        impact_audio_shuffled = []

        for item1, item2 in shuffle_together:
            impact_video_shuffled.extend(item1)
            impact_audio_shuffled.extend(item2)
        
        input_data_raw = impact_video_shuffled
        label_data = impact_audio_shuffled

    # training configuration
    input_data_selected = input_data_raw
    # crop and upsampling input data
    input_data_cropped = [crop_image(frame,224,224) for frame in input_data_selected]
    input_data = upsampling(input_data_cropped)
    label_data = label_data
    #print(np.asarray(label_data).shape)
    

    #plt.figure()
    #plt.imshow(np.asarray(label_data[0:45]).T)
    #plt.colorbar(orientation='vertical')
    #plt.show()
    #plt.savefig('test_label.png')
    
    
    model = cnn_rnn(batch_size=180, learn_rate=3e-6, keep_prob=1)
    with tf.Session() as sess:
        
        model.train(sess, if_restore, input_data, label_data, epoch=1001, check_point=check_point)
        #model.test(sess, input_data[100:280])
        #model.test(sess, input_data[3000:3045])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='option for continue training.')
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('check_point', nargs='?')
    args = parser.parse_args()
    main(if_shuffle=True, if_restore=args.resume, check_point=args.check_point)
