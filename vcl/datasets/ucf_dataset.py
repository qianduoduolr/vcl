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
from PIL import Image
import csv

from mmcv import scandir
import mmcv

from .base_dataset import BaseDataset
from .video_dataset import *
from .registry import DATASETS

from .pipelines import Compose
from vcl.utils import *


@DATASETS.register_module()
class UCF_dataset_rgb(Video_dataset_base):
    
    def __init__(self, rand_step=False, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.load_annotations()
        self.rand_step = rand_step

    def load_annotations(self):
        
        self.samples = []
        list_path = osp.join(self.list_path, f'{self.split}_list_01.csv')

        cols_name = ['duration_flow', 'duration_rgb', 'label', 'video_class',  'video_path',  'vname'] 
        with open(list_path) as f:
            f_csv = csv.DictReader(f, fieldnames=cols_name, delimiter=' ')
            rows = list(f_csv)
            for idx, sample in enumerate(rows):
                if idx == 0: continue

                video_path = os.path.join(self.root, sample['video_path'])
                num_frames = int(sample["duration_rgb"])

                if self.data_backend != 'lmdb':
                    raise NotImplementedError
                else:
                    if not os.path.exists(osp.join(video_path,'data.mdb')): continue

                    sample['frames_path'] = [os.path.join(video_path, self.filename_tmpl.format(i)) for i in range(1,num_frames+1)]
                
                sample['num_frames'] = len(sample['frames_path'])
                
                if sample['num_frames'] < self.clip_length * self.step: continue
                
                self.samples.append(sample)
        
        # self.samples = self.samples[:32]
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples)))
    
    
    def prepare_train_data(self, idx):

        sample = self.samples[idx]
        frames_path = sample['frames_path']
        num_frames = sample['num_frames']

        step = random.randint(1, self.step) if self.rand_step else self.step

        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step, mode=self.temporal_sampling_mode)
        
        # load frame
        if self.data_backend == 'lmdb':
            raise NotImplementedError
        else:
            frames = self._parser_rgb_lmdb_deprected(offsets, frames_path, self.clip_length, step=step)

        data = {
            'imgs': frames,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        } 

        return self.pipeline(data)