import numpy as np
import glob
import os.path as osp
import mmcv

year = '2017'
imset = ['train','val']

resolution = '480p'
root = '/home/lr/dataset/DAVIS'
list_path = '/home/lr/dataset/DAVIS/ImageSets'
anno_path = osp.join(root, 'Annotations', resolution)
frame_path = osp.join(root, 'JPEGImages', resolution)


for mode in imset:
    with open(f'davis{year}_{mode}_list.txt','a') as f:
        path = osp.join(list_path, year, mode+'.txt')
        with open(path, 'r') as r:
            videos_subset = r.readlines()
        for video in videos_subset:
            video = video.strip('\n')
            video_path = osp.join(frame_path, video)
            frame_num = len(glob.glob(osp.join(video_path, '*.jpg')))
            f.write(f'{video} {frame_num}' + '\n')



