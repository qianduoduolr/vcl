import numpy as np
import glob
import os.path as osp
import mmcv

year = '2018'
imset = ['train','test']

resolution = '480p'
root = '/home/lr/dataset/YouTube-VOS'


for mode in imset:
    anno_path = osp.join(root, mode, 'Annotations')
    frame_path = osp.join(root, mode, 'JPEGImages')   
    with open(f'youtube{year}_{mode}_list.txt','a') as f:
       
        videos_subset = glob.glob(osp.join(frame_path, '*'))
        for video in videos_subset:

            video_name = video.split('/')[-1]

            frame_num = len(glob.glob(osp.join(video, '*.jpg')))
            f.write(f'{video_name} {frame_num}' + '\n')







