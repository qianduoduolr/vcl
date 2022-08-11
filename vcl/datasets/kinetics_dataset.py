import copy
from operator import length_hint
import os.path as osp
from collections import defaultdict
from pathlib import Path
import glob
import os
import random
import pickle
from mmcv.fileio.io import load
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np
import cv2
import csv
from PIL import Image

from mmcv import scandir
import mmcv

from .base_dataset import BaseDataset
from .video_dataset import *
from .registry import DATASETS

from .pipelines import Compose
from vcl.utils import *


@DATASETS.register_module()
class Kinetics_dataset_rgb(Video_dataset_base):
    
    def __init__(self, rand_step=False, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.load_annotations()
        self.rand_step = rand_step

    def load_annotations(self):
        
        self.samples = []
        list_path = osp.join(self.list_path, f'{self.split}_list.csv')

        cols_name = ['duration_flow', 'duration_rgb', 'label', 'video_class',  'video_path',  'vname'] 
        with open(list_path) as f:
            f_csv = csv.DictReader(f, fieldnames=cols_name, delimiter=' ')
            rows = list(f_csv)
            for idx, sample in enumerate(rows):
                if idx == 0: continue

                if self.data_backend == 'raw_video':
                    video_path = os.path.join(self.root, sample['video_path'] + '.mp4')
                    if os.path.exists(video_path): 
                        sample['video_path'] = video_path
                    else:
                        continue
                else:
                    video_path = os.path.join(self.root, sample['video_path'])
                    try:
                        with open(osp.join(video_path, 'split.txt'), 'r') as f:
                            num_frames = int(f.readline().strip('\n'))

                        sample['frames_path'] = [os.path.join(video_path, self.filename_tmpl.format(i * 5)) for i in range(1,num_frames+1)]
                        sample['num_frames'] = len(sample['frames_path'])
                        if sample['num_frames'] <= self.clip_length * self.step: continue
                    except Exception as e: 
                        continue
                
                self.samples.append(sample)
        
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples)))
    
    
    def prepare_train_data(self, idx):
        
        step = random.randint(1, self.step) if self.rand_step else self.step
        sample = self.samples[idx]
         
        # load frame
        if self.data_backend == 'raw_video':
            data = {
                'filename': sample['video_path'],
                'start_index': self.start_index,
                'modality': 'RGB',
                'num_proposals':1
            }

        elif self.data_backend == 'lmdb':
            frames_path = sample.get('frames_path', None)
            num_frames = sample.get('num_frames', None)
            offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step, mode=self.temporal_sampling_mode)
            frames = self._parser_rgb_lmdb_deprected(offsets, frames_path, self.clip_length, step=step, name_idx=-3)

            data = {
                'imgs': frames,
                'modality': 'RGB',
                'num_clips': self.num_clips,
                'num_proposals':1,
                'clip_len': self.clip_length
            } 

        return self.pipeline(data)