import torch
import numpy as np
import os
import json
import pprint
import torch
from torch import nn, einsum

from decord import VideoLoader, VideoReader
from decord import cpu
from einops import rearrange, repeat, reduce

from mmpt.models import MMPTModel


def read_video():
    root = '/home/v-yifangxu/Desktop/datasets/TACoS/videos/'

    # s26-d26.avi   s37-d25.avi
    with open(root + 's26-d26.avi', 'rb') as f:
        # h, w = 1224, 1624
        # h, w = int(h // 4), int(w // 4)
        vr = VideoReader(f, ctx=cpu(0))
        # fps = vr.get_avg_fps()
        interval = 30

        video = vr.get_batch(list(range(0, len(vr), interval))).asnumpy()
        print(video.shape)

        video = torch.randn([4124, 1224, 1624, 3]).to('cuda')
        print(video.shape)

        video = torch.from_numpy(video).to('cuda')       # [T, H, W, C]
        print(video.shape)

        time = video.shape[0]
        fps = 30
        start_time = time % fps
        video = video[start_time:]

        video = rearrange(video, '(t fps) h w c -> 1 t fps h w c', fps=fps)     # [B, T, FPS, H, W, C]

        print(video.shape)

    return video


def read_feat():
    # 读取经过S3D的特征
    root = '/home/v-yifangxu/Desktop/fairseq/examples/MMPT/data/feat/feat_how2_s3d/'
    file_list = os.listdir(root)

    feat = np.load(root + 's26-d26.npy')
    feat = torch.from_numpy(feat).float()
    feat = rearrange(feat, 'n d -> 1 n d')
    print("shape:", feat.shape)

    if False:
        for f in file_list:
            if f.split('.')[1] != 'npy':
                continue
            data = np.load(root + f)
            print("shape:", data.shape)

    return feat


def read_txt():
    root = '/home/v-yifangxu/Desktop/fairseq/examples/MMPT/data/how2/'
    train_file = 'TACoS_samples.txt'

    with open(root + train_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        text_list = []

        for line in lines:
            video_id = line.split('.')[0]

            if video_id == 's26-d26':
                text = line.split(':')[1].split('#')[:-1]
                text_list.extend(text)

    res = ''
    for t in text_list:
        res += t + ' '
    return res[:-1]


model, tokenizer, aligner = MMPTModel.from_pretrained("projects/retri/videoclip/how2.yaml")
model.eval()

# [B, T, FPS, H, W, C]  (VideoCLIP is trained on 30 fps of s3d)
# video_frames = read_video()
# video_feat = read_feat()
video_feat = torch.randn(1, 2, 30, 224, 224, 3)
# text = read_txt()
# print('len(text): {}'.format(len(text)))

caps, cmasks = aligner._build_text_seq(
    tokenizer('text', add_special_tokens=False)["input_ids"]
)

# print('caps.shape: {}'.format(caps.shape))
# caps, cmasks = torch.randint(0, 100, [1500]), torch.randn([1500])
caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1      [bsz, N]

with torch.no_grad():
    output = model(video_feat, caps, cmasks, return_score=True)
print(output["score"])  # dot-product


