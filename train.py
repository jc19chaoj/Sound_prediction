import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import data_prepare
from model_alexnet import cnn_rnn
from random import shuffle
from itertools import chain

LOGDIR = './training_log'

def upsampling(inputs):
    input_data = []
    for i in range(len(inputs)):
        for j in range(3):
            input_data.append(inputs[(j+i)%len(inputs)])
    return input_data

def crop_image(img,cropx,cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]

def main_test(if_shuffle=False, if_restore=False):
    dp = data_prepare(train_size=10)
    dp.video_cap()
    dp.prediction_frames()
    dp.load_audio()

    input_data_raw = dp.impact_frames
    label_data = dp.sound_features
    #test_data = dp.impact_videos[0][0]
    test_data = dp.impact_frames[0:15]
    test_data = [val for val in test_data for _ in range(3)] # up-sampling input data by 3
    
    if if_shuffle:
        #impact_videos_shuffled = list(chain.from_iterable(dp.impact_videos))
        impact_video_shuffled = [input_data_raw[i:i+15] for i in range(len(input_data_raw)//15)]
        impact_audio_shuffled = [label_data[i:i+45] for i in range(len(label_data)//45)]
        
        shuffle_together = list(zip(impact_video_shuffled, impact_audio_shuffled))
        shuffle(shuffle_together)
        
        impact_video_shuffled = [item for item,_ in shuffle_together]
        impact_audio_shuffled = [item for _,item in shuffle_together]
        
        input_data_raw = list(chain.from_iterable(impact_video_shuffled))
        label_data = list(chain.from_iterable(impact_audio_shuffled))
                
    # test
    #input_data = [tf.zeros([300,300,3]) for i in range(30)]
    #label_data = [tf.zeros([42,]) for i in range(30)]
    print(len(label_data),len(label_data[0]),label_data[0])
    print(len(input_data_raw),input_data_raw[0].shape)
    print(input_data_raw[0].shape)
    
    # training configuration
    input_data_selected = input_data_raw#[0:45]
    # up-sampling input data by 3
    #input_data = [crop_image(val,224,224) for _ in range(3) for val in input_data] 
    input_data_cropped = [crop_image(frame,224,224) for frame in input_data_selected]
    input_data = upsampling(input_data_cropped)
    label_data = label_data#[0:135]
    #print(tf.convert_to_tensor(np.asarray(label_data)))
    
    plt.figure()
    plt.imshow(np.asarray(np.asarray(label_data[0:45]).T))
    plt.colorbar(orientation='vertical')
    #plt.show()
    plt.savefig('test_label.png')
    
    
    model = cnn_rnn(batch_size=45, learn_rate=3e-5, keep_prob=1)
    with tf.Session() as sess:
        
        model.train(sess, if_restore, input_data, label_data, epoch=100, crop_inputs=False)
        model.test(sess, input_data[0:45])
    
if __name__ == "__main__":
    main_test(if_shuffle=True, if_restore=False)
