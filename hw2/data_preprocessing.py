import numpy as np
import os
import sys
from os import listdir
import json
import pickle
import io

def build_dictionary(sentences,min_count):
    word_counts = {}
    sentence_count=0
    for sentence in sentences:
        sentence_count+=1
        for word in sentence.lower().split():
            word_counts[word] = word_counts.get(word,0)+1
    dictionary = [word for word in word_counts if word_counts[word]>=min_count]

    print("Filtered words from %d to %d with min count [%d]" % (len(word_counts),len(dictionary),min_count))

    index2word = {
        0:'<pad>',
        1:'<bos>',
        2:'<eos>',
        3:'<unk>'
    }
    word2index = {
        '<pad>':0,
        '<bos>':1,
        '<eos>':2,
        '<unk>':3
    }

    for index,word in enumerate(dictionary):
        index2word[index+4] = word
        word2index[word] = index+4

    return word2index, index2word, dictionary

def filter_token(string):
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    for c in filters:
        string.replace(c,'')
    return string

if __name__=="__main__":
    video_feat_folder = sys.argv[1]
    training_label_json_file = sys.argv[2]

    video_feat_filenames = listdir(video_feat_folder)
    video_feat_filepaths = [(video_feat_folder + filename) for filename in video_feat_filenames]

    #remove '.npy' from the filenames
    video_IDs = [filename[:-4] for filename in video_feat_filenames]
    #print(video_IDs)

    video_feat_dict = {} #store video features
    for path in video_feat_filepaths:
        video_feat = np.load(path)
        video_ID = path[:-4].replace(video_feat_folder,"")
        video_feat_dict[video_ID] = video_feat

    #store video captions
    video_captions = json.load(open(training_label_json_file,'r'))
    video_caption_dict = {}
    caption_corpus = []

    for video in video_captions:
        filtered_captions = [filter_token(sentence) for sentence in video["caption"]]
        video_caption_dict[video["id"]] = filtered_captions
        caption_corpus += filtered_captions


    word2index, index2word, dictionary = build_dictionary(caption_corpus,min_count=3)

    with open('./word2index.obj','wb') as file:
        pickle.dump(word2index,file,protocol=pickle.HIGHEST_PROTOCOL)
    with open('./index2word.obj','wb') as file:
        pickle.dump(index2word,file,protocol=pickle.HIGHEST_PROTOCOL)
    with open('video_IDs.obj', 'wb') as file:
        pickle.dump(video_IDs,file,protocol=pickle.HIGHEST_PROTOCOL)
    with open('video_caption_dict.obj', 'wb') as file:
        pickle.dump(video_caption_dict,file,protocol=pickle.HIGHEST_PROTOCOL)
    """with open('video_feat_dict.obj', 'wb') as file:
        pickle.dump(video_feat_dict,file,protocol=pickle.HIGHEST_PROTOCOL)"""