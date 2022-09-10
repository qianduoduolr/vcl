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
from glob import glob
from PIL import Image

from mmcv import scandir
import mmcv

from .base_dataset import BaseDataset
from .video_dataset import *
from .registry import DATASETS
from .pipelines.my_aug import ClipRandomSizedCrop, ClipMotionStatistic, ClipRandomHorizontalFlip

from .pipelines import Compose
from vcl.utils import *


@DATASETS.register_module()
class YFCC_dataset_rgb(Video_dataset_base):
    def __init__(self, data_prefix, 
                       rand_step=False,
                       **kwargs
                       ):
        super().__init__(**kwargs)

        self.data_prefix = data_prefix
        self.rand_step = rand_step
        self.load_annotations()

    def __len__(self):
        return len(self.samples)

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.data_prefix['RGB'])
        images = glob(osp.join(self.video_dir, '*/*'))

        data = defaultdict(list)

        for image in images:
            vname = osp.basename(image).split('_')[0]
            data[vname].append(image)
        
        for vname, frames_path in data.items():
            sample = dict()
            sample['frames_path'] = sorted(frames_path)
            sample['num_frames'] = len(sample['frames_path'])
            if sample['num_frames'] <= self.clip_length * self.step:
                continue
        
            self.samples.append(sample)
        
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples)))

    
    
    def prepare_train_data(self, idx):
        
        if self.data_backend == 'lmdb' and self.env == None and self.txn == None:
            self._init_db(self.video_dir)

        sample = self.samples[idx]
        frames_path = sample['frames_path']
        num_frames = sample['num_frames']
        
        step = random.randint(1, self.step) if self.rand_step else self.step

        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, step, mode=self.temporal_sampling_mode)

        # load frame
        if self.data_backend == 'raw_frames':
            frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=step)
        elif self.data_backend == 'lmdb':
            frames = self._parser_rgb_lmdb(self.txn, offsets, frames_path, self.clip_length, step=step)

        data = {
            'imgs': frames,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        } 

        return self.pipeline(data)