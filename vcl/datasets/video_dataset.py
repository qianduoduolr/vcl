# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset
from .base_dataset import BaseDataset

import random
import mmcv
import numpy as np


class Video_dataset_base(BaseDataset):
    def __init__(self, root,  
                       list_path, 
                       num_clips=1,
                       clip_length=1,
                       step=1,
                       pipeline=None, 
                       test_mode=False,
                       filename_tmpl='{:05d}.jpg',
                       temporal_sampling_mode='random',
                       split='train'
                       ):
        super().__init__(pipeline, test_mode)

        self.clip_length = clip_length
        self.num_clips = num_clips
        self.step = step
        self.list_path = list_path
        self.root = root
        self.filename_tmpl = filename_tmpl
        self.temporal_sampling_mode = temporal_sampling_mode
        self.split = split

    def temporal_sampling(self, num_frames, num_clips, clip_length, step, mode='random'):
            
        if mode == 'random':
            offsets = [ random.randint(0, num_frames-clip_length * step) for i in range(num_clips) ]
        elif mode == 'distant':
            length_ext = num_frames / num_clips
            offsets = np.floor(np.arange(num_clips) * length_ext + np.random.uniform(low=0.0, high=length_ext, size=(num_clips))).astype(np.uint8)
        elif mode =='mast':
            short_term_interval = 2
            offsets_long_term = [0,1]
            short_term_start = random.randint(2, num_frames-clip_length * step - (num_clips-2) * short_term_interval )
            offsets_short_term = list([ short_term_start+i*short_term_interval for i in range(num_clips-2)])
            offsets = offsets_long_term + offsets_short_term
        
        return offsets

    def _parser_rgb_rawframe(self, offsets, frames_path, clip_length, step=1, flag='color', backend='cv2'):
        """read frame"""
        frame_list_all = []
        for offset in offsets:
            for idx in range(clip_length):
                frame_path = frames_path[offset + idx]
                frame = mmcv.imread(frame_path, backend=backend, flag=flag, channel_order='rgb')
                frame_list_all.append(frame)
        return frame_list_all

    def _parser_rgb_lmdb(self, offsets, frames_path, clip_length, step=1, flag='color', backend='cv2'):
        """read frame"""
        frame_list_all = []
        for offset in offsets:
            frame_list = []
            for idx in range(clip_length):
                frame_path = frames_path[offset + idx]
                frame = mmcv.imread(frame_path, backend=backend, flag=flag, channel_order='rgb')
                frame_list.append(frame)
            frame_list_all.append(frame_list)
        return frame_list_all if len(frame_list_all) >= 2 else frame_list_all[0]