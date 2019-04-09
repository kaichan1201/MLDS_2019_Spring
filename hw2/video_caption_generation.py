import numpy as mp
import json
import pickle
import random
import torch
import torch.utils.data as Data
import data_preprocessing

if __name__ == "__main__":
    np.random.seed(9487)

    video_feat_folder = 'MLDS_hw2_1_data/training_data/feat/'
    training_label_json_file = 'MLDS_hw2_1_data/training_label.json'

    video_feat_filenames = listdir(video_feat_folder)
    video_feat_filepaths = [(video_feat_folder + filename) for filename in video_feat_filenames]

    with open('video_IDs.obj') as file:
        video_IDs = pickle.load(file)
    with open('word2index.obj') as file:
        word2index = pickle.load(file)
    with open('index2word.obj') as file:
        index2word = pickle.load(file)
    with open('video_caption_dict.obj') as file:
        video_caption_dict = pickle.load(file)
        
    video_feat_dict = {}
    for filepath in video_feat_filepaths:
        video_feat = np.load(filepath)
        video_ID = filepath[: -4].replace(video_feat_folder, "")
        video_feat_dict[video_ID] = video_feat

    x = video_feat_dict.values()
    y = [my_caption[random.random() % len(my_caption)] for my_caption in video_caption_dict.values()]