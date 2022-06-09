#!/bin/bash


python scripts/video_feature_extractor/extract.py \
    --vdir /home/v-yifangxu/Desktop/datasets/TACoS/videos \
    --fdir data/feat/feat_how2_s3d \
    --type=s3d --num_decoding_thread=1 \
    --batch_size 1 --half_precision 1
