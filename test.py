import numpy as np
import os
import json
import pprint
import torch
from torch import nn, einsum

from decord import VideoLoader, VideoReader
from decord import cpu, gpu
from einops import rearrange, repeat, reduce


def print_feature_shape():
    # 查看所有特征的大小
    root = '/home/v-yifangxu/Desktop/fairseq/examples/MMPT/data/feat/feat_how2_s3d/'
    file_list = os.listdir(root)

    for f in file_list:
        if f.split('.')[1] != 'npy':
            continue
        data = np.load(root + f)
        print("shape:", data.shape)


def read_json():
    root = '/home/v-yifangxu/Desktop/fairseq/examples/MMPT/data/how2/'
    with open(root + 'caption.json', 'r', encoding="utf-8") as f:
        res = json.load(f)

    print(res[0])


def read_txt():
    root = '/home/v-yifangxu/Desktop/fairseq/examples/MMPT/data/how2/'
    train_file = 'TACoS_samples.txt'

    with open(root + train_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()

        text_dict = {}

        for line in lines:
            video_id = line.split('.')[0]
            start = line.split('_')[1]
            end = line.split('_')[2].split(':')[0]
            text = line.split(':')[1].split('#')[:-1]
            start_list, end_list, text_list = [], [], []

            for t in text:
                start_list.append(int(start))
                end_list.append(int(end))
                text_list.append(t)

            text_dict[video_id] = {
                'start': start_list,
                'end': end_list,
                'text': text
            }

    json_file = 'caption.json'
    with open(root + json_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(text_dict))

    # 验证是否写入
    with open(root + json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(data.keys())


def read_video():
    root = '/home/v-yifangxu/Desktop/datasets/TACoS/videos/'

    with open(root + 's13-d21.avi', 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
        # fps = vr.get_avg_fps()
        fps = 30

        video = torch.from_numpy(vr[:].asnumpy()).float()                       # [T, H, W, C]

        time = video.shape[0]
        start_time = time % fps
        video = video[start_time:]

        video = rearrange(video, '(t fps) h w c -> 1 t fps h w c', fps=fps)     # [B, T, FPS, H, W, C]

read_video()