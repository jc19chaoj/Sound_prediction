import numpy as np
import tensorflow as tf
from data import *
from config import *
from model_alexnet import *
import argparse


def normalize(input_array):
    return (input_array-np.min(input_array))/(np.max(input_array)-np.min(input_array))

class audio_synthesize:

    def __init__(self, label_data):
        self.label_data = label_data
        self.filenames = [os.path.join(audio_path, filename) for filename in sorted(os.listdir(audio_path))]
        self.sound_filename = None

    def synthesize(self, predict):
        min_l2_loss = np.inf
        print(predict)
        #print(self.label_data)
        for key,value in self.label_data.items():
            for feature in value[0]:
                #print("predict:", predict.shape, feature.shape)
                l2_loss = np.linalg.norm((normalize(predict[:,:,:]) - normalize(feature[:,:])))
               # print("l2_loss:", l2_loss)

                if l2_loss < min_l2_loss:
                    min_l2_loss = l2_loss
                    self.sound_filename = key


def main(check_point, train_size=[0,15], test_video=0):
    sd = data_prepare(train_size=[0,100])
    sd.load_audio(test=test_video)
    generate_audio = audio_synthesize(sd.sound_database)

    dp = data_prepare(train_size=train_size, impact_size=1)
    dp.video_cap()
    dp.prediction_frames(test=test_video)

    test_video = dp.impact_videos[test_video][0]
   # test_video = [np.random.rand(600,300,3) for _ in range(15)]
    test_data = upsampling(test_video)
    test_data = [crop_image(frame,224,224) for frame in test_data]
    # print(generate_audio.label_data[generate_audio.filenames[0]][0].shape)

    model = cnn_rnn(keep_prob=1)
    with tf.Session() as sess:
        #print(test_data)
        predict = model.inference(sess, test_data, check_point)
        model.test(sess, test_data)

    generate_audio.synthesize(predict)

    print("Synthesized sound file:", generate_audio.sound_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("check_point")
    parser.add_argument("test_video", type=int)
    args = parser.parse_args()
    main(check_point=args.check_point, test_video=args.test_video)
