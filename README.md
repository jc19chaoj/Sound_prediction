# Sound_prediction

Download link for:

  video files (~50 GB): http://vis.csail.mit.edu/vis-data.zip
  
  audio files (~1 GB): http://vis.csail.mit.edu/vis-sfs.zip

  pre-trained AlexNet (~250 MB): http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy


Data preprocess:

  Create four folders: source_code, audio_mat, audio_txt, video_thumb, and source_code 
  
  Put:
    ".mat" files into "audio_mat" folder.
    ".txt" files into "audio_txt" folder.
    "xxx_denoised_thumb.mp4" files into "video_thumb" folder.
    all Python code and AlexNet model into "source_code" folder.
    

Training:

  Run "python train.py" to train from scratch. A checkpoint model will be created every 50 epoch.
  Run "python train.py --resume <check_point_number, e.g., 50>" to resume training.


Inference:

  Run "python synthesize.py <check_point_number> <video_id>" to generate audio from a short video based on the training result.
