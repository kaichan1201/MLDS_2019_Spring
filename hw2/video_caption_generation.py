import numpy as mp
import json
import pickle
import random
import torch
import torch.utils.data as Data
import data_preprocessing
import s2s

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
    pre_y = [captions[random.random() % len(captions)].split() for captions in video_caption_dict.values()]
    y = []
    for sentence in pre_y:
        tokens = [word2index[word] for word in sentence]
        y.append(tokens)
    
    torch_dataset = Data.TensorDataset(data_tensor = x, target_tensor = y)
    
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = 64,
        shuffle = True,
        num_workers=2,
    )

enc = Encoder()
dec = Decoder()

model = Seq2Seq(encoder,decoder).to(device)

#count param
def count_parametes(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

optimizer = optim.Adam(model.parameters(),lr= LR)
