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

from mmcv import scandir
import mmcv

from .base_dataset import BaseDataset
from .video_dataset import *
from .registry import DATASETS
from .pipelines.my_aug import ClipRandomSizedCrop, ClipMotionStatistic, ClipRandomHorizontalFlip

from .pipelines import Compose
from vcl.utils import *


@DATASETS.register_module()
class VOS_youtube_dataset_rgb(Video_dataset_base):
    def __init__(self, data_prefix, 
                       rand_step=False,
                       year='2018',
                       **kwargs
                       ):
        super().__init__(**kwargs)

        self.data_prefix = data_prefix
        self.year = year
        self.rand_step = rand_step
        self.load_annotations()

    def __len__(self):
        return len(self.samples)

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        list_path = osp.join(self.list_path, f'youtube{self.year}_{self.split}.json')
        data = mmcv.load(list_path)
        
        for vname, frames in data.items():
            sample = dict()
            sample['frames_path'] = []
            for frame in frames:
                sample['frames_path'].append(osp.join(self.video_dir, vname, frame))
                
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

        offsets = [0]

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
    
    
@DATASETS.register_module()
class VOS_youtube_dataset_rgb_V2(VOS_youtube_dataset_rgb):    
    
    def prepare_train_data(self, idx):

        sample = self.samples[idx]
        frames_path = sample['frames_path']
        num_frames = sample['num_frames']

        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step, mode=self.temporal_sampling_mode)
        offsets_ = self.temporal_sampling(num_frames, 1, self.num_clips, self.step, mode=self.temporal_sampling_mode)
        
        # load frame
        if self.data_backend == 'raw_frames':
            frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=self.step)
            frames_ = self._parser_rgb_rawframe(offsets_, frames_path, self.num_clips, step=self.step)
        elif self.data_backend == 'lmdb':
            frames = self._parser_rgb_lmdb(offsets, frames_path, self.clip_length, step=self.step)
            frames_ = self._parser_rgb_lmdb(offsets_, frames_path, self.clip_length, step=self.step)
        

        data = {
            'imgs': frames_,
            'imgs_spa_aug': frames,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        } 

        return self.pipeline(data)





@DATASETS.register_module()
class VOS_youtube_dataset_rgb_flow(VOS_youtube_dataset_rgb):

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        self.flow_dir = osp.join(self.root, self.year, self.data_prefix['FLOW'])
        
        list_path = osp.join(self.list_path, f'youtube{self.year}_{self.split}.json')
        data = mmcv.load(list_path)
        
        for vname, frames in data.items():
            sample = dict()
            sample['frames_path'] = []
            sample['flows_path'] = []
            
            for idx, frame in enumerate(frames[:-1]):
                if idx < len(frames) -1:
                    sample['frames_path'].append(osp.join(self.video_dir, vname, frame))
                    sample['flows_path'].append(osp.join(self.flow_dir, vname, frame))
                
            sample['num_frames'] = len(sample['frames_path'])
            if sample['num_frames'] < self.clip_length * self.step:
                continue
        
            self.samples.append(sample)
        
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples)))
        
    def prepare_train_data(self, idx):
        
        if self.data_backend == 'lmdb' and self.env == None and self.txn == None:
            self._init_db(self.video_dir)

        sample = self.samples[idx]
        frames_path = sample['frames_path']
        flows_path = sample['flows_path']
        num_frames = sample['num_frames']
        
        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, self.step, mode=self.temporal_sampling_mode)
        
        # load frame
        if self.data_backend == 'raw_frames':
            frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=self.step)
        elif self.data_backend == 'lmdb':
            frames = self._parser_rgb_lmdb(self.txn, offsets, frames_path, self.clip_length, step=self.step)

        flows = self._parser_rgb_rawframe(offsets, flows_path, self.clip_length, step=self.step)
        flows = [ cv2.resize(flow, frames[0].shape[:2][::-1]) for flow in flows ]
        
        
        data = {
            'imgs': frames,
            'flows': flows,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        } 

        return self.pipeline(data)