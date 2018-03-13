import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import data_prepare
from model import cnn_rnn

def main_test(if_shuffle=False):
    dp = data_prepare(train_size=10)
    dp.video_cap()
    dp.prediction_frames()
    dp.load_audio()

    input_data = dp.impact_frames
    label_data = dp.sound_features
    #test_data = dp.impact_videos[0][0]
    test_data = dp.impact_frames[0:15]
    test_data = [val for val in test_data for _ in range(3)] # up-sampling input data by 3
    
    if if_shuffle:
        #impact_videos_shuffled = list(chain.from_iterable(dp.impact_videos))
        impact_video_shuffled = [input_data[i:i+15] for i in range(len(input_data)//15)]
        impact_audio_shuffled = [label_data[i:i+45] for i in range(len(label_data)//45)]
        
        shuffle_together = list(zip(impact_video_shuffled, impact_audio_shuffled))
        shuffle(shuffle_together)
        
        impact_video_shuffled = [item for item,_ in shuffle_together]
        impact_audio_shuffled = [item for _,item in shuffle_together]
        
        input_data = list(chain.from_iterable(impact_video_shuffled))
        label_data = list(chain.from_iterable(impact_audio_shuffled))
                
    # test
    #input_data = [tf.zeros([300,300,3]) for i in range(30)]
    #label_data = [tf.zeros([42,]) for i in range(30)]
    print(len(label_data),len(label_data[0]),label_data[0])
    print(len(input_data),input_data[0].shape)
    print(input_data[6059].shape)
    
    # training configuration
    input_data = input_data[0:15]
    input_data = [val for val in input_data for _ in range(3)] # up-sampling input data by 3
    label_data = label_data[0:45]
    #print(tf.convert_to_tensor(np.asarray(label_data)))
    
    plt.figure()
    plt.imshow(np.asarray(np.asarray(label_data).T))
    plt.colorbar(orientation='vertical')
    plt.show()
    
    model = cnn_rnn(batch_size=45, learn_rate=0.001, keep_prob=1)

    model.train(input_data, label_data, input_data, epoch=1000)
    
    #model.test(test_data)
    
if __name__ == "__main__":
    main_test(if_shuffle=False)
