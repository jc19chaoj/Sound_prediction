import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import scipy.io as spio
#%matplotlib

class data_prepare:
    def __init__(self, train_size=10, video_path="../video_thumb", impact_path="../audio_txt", audio_path="../audio_mat"):
        # path of video files
        self.video_path = video_path
        self.impact_path = impact_path
        self.audio_path = audio_path
        self.videos = []
        self.impact_frames = []
        self.impact_videos = []
        self.audio_list = []
        self.sound_features = []
        self.train_size = train_size

    def video_cap(self,test=None):
        """capture all videos in video dir"""   
        cap_list = []
        i = 0
        for filename in sorted(os.listdir(self.video_path)):
            if i<self.train_size:
                filename = os.path.join(self.video_path, filename)
                cap = cv2.VideoCapture(filename)
                cap_list.append(cap)
                i += 1
            else:
                break

        # store all frames into a list
        for i in range(len(cap_list)):
            frames = []
            while True:
                ret, frame = cap_list[i].read()
                if cv2.waitKey(1) & 0xFF == ord('q') or ret==False:
                    cap_list[i].release()
                    break
                frames.append(frame)
            self.videos.append(frames)
        
        # play a random video in the list
        if test!=None:
            for frame in self.videos[test]:
                cv2.imshow('frame',frame)
                cv2.waitKey(1)
            cv2.destroyAllWindows()
            
            
    def prediction_frames(self,test=None):
        """prepare the input frames for prediction task"""
        onset_list = []
        impact_onsets = []

        i = 0
        for filename in sorted(os.listdir(self.impact_path)):
            if i<self.train_size:
                filename = os.path.join(self.impact_path, filename)
                impact_onset = pd.read_csv(filename, sep=" ", header=None)
                onset_list.append(impact_onset)
                i += 1
            else:
                break

        for onset in onset_list:
            impact_onsets.append(round(onset[0]*29.97).tolist())

        for i in range(len(impact_onsets)):
            impact_frames_temp = []
            for onset in impact_onsets[i]:
                onset = int(onset)
                self.impact_frames.extend(self.videos[i][onset-7:onset+8])
                impact_frames_temp.append(self.videos[i][onset-7:onset+8])
            self.impact_videos.append(impact_frames_temp)

        if test!=None:
            # play a random clip
            for frame in self.impact_videos[test][0]:
                cv2.imshow('frame',frame)
                cv2.waitKey(100)
            cv2.destroyAllWindows()

            
    def load_audio(self,test=None):
        """prepare the desired outputs for prediction task"""   
        i = 0
        for filename in sorted(os.listdir(self.audio_path)):
            if i<self.train_size:
                filename = os.path.join(self.audio_path, filename)
                mat = spio.loadmat(filename, squeeze_me=True)
                mat = mat['sfs']
                
                self.audio_list.append(mat)
                i += 1
                
                for j in range(mat.shape[0]):
                    self.sound_features.extend(mat[j])
            else:
                break

        if test!=None:
            # plot a random sound feature
            plt.figure()
            plt.imshow(self.audio_list[test][0].T)
            plt.colorbar(orientation='vertical')
            plt.show()
