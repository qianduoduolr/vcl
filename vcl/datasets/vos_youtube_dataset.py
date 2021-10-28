import copy
import os.path as osp
from collections import defaultdict
from pathlib import Path
import glob
import os
import random
import torch
import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image

from mmcv import scandir
import mmcv

from .base_dataset import BaseDataset
from .registry import DATASETS
from .pipelines.my_aug import aug_heavy
from .vos_davis_dataset import VOS_dataset_base

from .pipelines import Compose

MAX_OBJECT_NUM_PER_SAMPLE = 5


@DATASETS.register_module()
class VOS_youtube_dataset_stm(VOS_dataset_base):

    def __init__(self, root,  
                       list_path, 
                       sample_type='stm', 
                       pipeline=None, 
                       test_mode=False
                       ):
        super().__init__(root, list_path, pipeline, test_mode)

        self.K = 11
        self.skip = 0
        self.sample_type = sample_type
        self.aug = aug_heavy()
    
    def load_annotations(self):
        
        self.samples = []

        mask_dir = osp.join(self.root, 'Annotations')
        video_dir = osp.join(self.root, 'JPEGImages')
        list_path = osp.join(self.list_path, 'youtube2018_train_list.txt') if not self.test_mode else  osp.join(self.list_path, 'youtube2018_test_list.txt')

        with open(list_path, 'r') as f:
            for line in f.readlines():
                sample = dict()
                vname, num_frames = line.strip('\n').split()
                sample['frames_path'] = sorted(glob.glob(osp.join(video_dir, vname, '*.jpg')))
                sample['masks_path'] = sorted(glob.glob(osp.join(mask_dir, vname, '*.png')))
                sample['vname'] = vname
                sample['num_frames'] = int(num_frames)
                self.samples.append(sample)
        
        # meta data
        self.meta_data = mmcv.load(osp.join(self.list_path, 'generated_frame_wise_meta.json'))

    def change_skip(self,f):
        self.skip = f

    def prepare_train_data_stm(self, index):

        sample = self.samples[index]
        num_frames = sample['num_frames']
        masks_path = sample['masks_path']
        frames_path = sample['frames_path']
       
        # sample on temporal
        n1 = random.sample(range(0,num_frames - 2),1)[0]
        n2 = random.sample(range(n1 + 1,min(num_frames - 1,n1 + 2 + self.skip)),1)[0]
        n3 = random.sample(range(n2 + 1,min(num_frames,n2 + 2 + self.skip)),1)[0]
        offsets = [n1,n2,n3]
        num_object = 0
        ob_list = []
        
        # load frames and masks
        frames = self._parser_rgb_jpg_cv(offsets, frames_path)
        masks = self._parser_mask(offsets, masks_path)

        # apply augmentations
        frames_,masks_ = self.aug(frames,masks)

        for f in range(len(frames)):
            masks_[f],num_object,ob_list = self.mask_process(masks_[f],f,num_object,ob_list)
            
        
        Fs = torch.from_numpy(np.transpose(np.stack(frames_, axis=0), (3, 0, 1, 2))).float()
        Ms = torch.from_numpy(self.All_to_onehot(np.stack(masks_, axis=0))).float()

        if num_object == 0:
            num_object += 1
            
        num_objects = torch.LongTensor([num_object])

        data = {
            'Fs': Fs,
            'Ms': Ms,
            'num_objects':num_objects
        }

        return data

    def prepare_train_data(self, idx):

        data_sampler = getattr(self, f'prepare_train_data_{self.sample_type}')

        return data_sampler(idx)

@DATASETS.register_module()
class VOS_youtube_dataset_pixel(VOS_dataset_base):

    def __init__(self, root,  
                       list_path, 
                       sample_type='pixel', 
                       pipeline=None, 
                       test_mode=False
                       ):
        super().__init__(root, list_path, pipeline, test_mode)
        
        self.K = 11
        self.sample_type = sample_type
        self.pipeline = Compose(pipeline)
        self.load_annotations()

    def load_annotations(self):
        self.samples = []

        mask_dir = osp.join(self.root, 'Annotations')
        video_dir = osp.join(self.root, 'JPEGImages')
        list_path = osp.join(self.list_path, 'youtube2018_train_list.txt') if not self.test_mode else  osp.join(self.list_path, 'youtube2018_test_list.txt')

        # meta data
        self.meta_data = mmcv.load(osp.join(self.list_path, 'generated_frame_wise_meta.json'))

        with open(list_path, 'r') as f:
            for line in f.readlines():
                obj_frames = defaultdict(list)
                sample = defaultdict(dict)
                vname, num_frames = line.strip('\n').split()

                video_info = self.meta_data[vname]

                for frame_name, objs in video_info.items():
                    for obj in objs:
                        obj_frames[obj[1]].append(frame_name)
                for obj, frames in obj_frames.items():
                    if len(frames) >= 3:
                        sample[obj]['frames_path'] = list([osp.join(video_dir, vname, f'{f}.jpg') for f in frames])
                        sample[obj]['masks_path'] = list([osp.join(mask_dir, vname, f'{f}.png') for f in frames])
                        sample[obj]['vname'] = vname
                
                if len(sample) > 0:
                    self.samples.append(sample)
        

    def obj_mask_process(self, frame_obj_masks, obj_num):
        
        random_flag = False
        masks = []
        for i,frame_obj_mask in enumerate(frame_obj_masks):
            obj_mask = (frame_obj_mask == obj_num).astype(np.uint8)
            value = obj_mask.sum()
            if value == 0: random_flag =True
            masks.append(obj_mask)
        
        if random_flag:
            rx = random.randint(0, masks[0].shape[0] - 1)
            ry = random.randint(0, masks[0].shape[0] - 1)
            for i in range(len(masks)):
                masks[i][rx, ry] = 1
            
        return np.array(masks)

    def prepare_train_data_pixel(self, index):
        
        sample = self.samples[index]
        obj_list = list(sample.keys())

        # random select a object
        obj_idx = random.sample(obj_list, 1)[0]

        sample_obj = sample[obj_idx]

        masks_path = sample_obj['masks_path']
        frames_path = sample_obj['frames_path']
        vname = sample_obj['vname']

        num_frames = len(frames_path)

        # sample on temporal
        n1 = random.randint(0, num_frames-2)
        offsets = [n1,n1+1]

        num_object = 0
        ob_list = []
        frame_obj_masks_ = []
        
        # load frames and masks
        frames = self._parser_rgb_jpg_cv(offsets, frames_path)
        masks = self._parser_mask(offsets, masks_path)

        # apply augmentations
        results = dict(images=frames, labels=masks, obj_num=obj_idx)
        results = self.pipeline(results)
        frames_, masks_ = results['images'], results['labels']

        # post processing for frames and masks
        for m in masks_:
            frame_obj_masks_.append(cv2.resize(m, (32,32), interpolation=cv2.INTER_NEAREST).astype(np.uint8))

        for f in range(len(frames)):
            masks_[f],num_object,ob_list = self.mask_process(masks_[f],f,num_object,ob_list)
            
        
        Fs = torch.from_numpy(np.transpose(np.stack(frames_, axis=0), (3, 0, 1, 2))).float()
        Ms = torch.from_numpy(self.All_to_onehot(np.stack(masks_, axis=0))).float()

        Ms_obj = self.obj_mask_process(frame_obj_masks_, obj_idx)
        Ms_obj = torch.from_numpy(Ms_obj)

        if num_object == 0:
            num_object += 1
            
        num_objects = torch.LongTensor([num_object])

        data = {
            'Fs': Fs,
            'Ms': Ms,
            'Ms_obj': Ms_obj,
            'num_objects':num_objects
        }

        return data

    def prepare_train_data_pair(self, index):

        sample = self.samples[index]
        masks_path = sample_obj['masks_path']
        frames_path = sample_obj['frames_path']
        vname = sample_obj['vname']

        num_frames = len(frames_path)

        # sample on temporal
        offsets = list([random.randint(0, num_frames-1) for i in range(2)])

        frames = self._parser_rgb_jpg_pillow(offsets, frames_path)
        results = dict(images=frames)

        return self.pipeline(results)



    def prepare_train_data(self, idx):

        data_sampler = getattr(self, f'prepare_train_data_{self.sample_type}')

        return data_sampler(idx)


@DATASETS.register_module()
class VOS_youtube_dataset_mlm(VOS_dataset_base):
    def __init__(self, root,  
                       list_path, 
                       data_prefix, 
                       mask_ratio=0.15,
                       clip_length=3,
                       vq_size=32,
                       pipeline=None, 
                       test_mode=False,
                       split='train'
                       ):
        super().__init__(root, list_path, pipeline, test_mode, split)


        self.list_path = list_path
        self.root = root
        self.data_prefix = data_prefix

        self.load_annotations()
        self.clip_length = clip_length
        self.vq_res = vq_size
        self.mask_ratio = mask_ratio


    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.data_prefix, self.split, 'JPEGImages')
        list_path = osp.join(self.list_path, f'youtube{self.data_prefix}_train_list.txt')

        with open(list_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                sample = dict()
                vname, num_frames = line.strip('\n').split()
                sample['frames_path'] = sorted(glob.glob(osp.join(self.video_dir, vname, '*.jpg')))
                sample['num_frames'] = int(num_frames)
                self.samples.append(sample)
    
    def prepare_test_data(self, idx):
        pass

    def prepare_train_data(self, idx):
        
        sample = self.samples[idx]
        frames_path = sample['frames_path']
        num_frames = sample['num_frames']

        offset = [ random.randint(0, num_frames-3)]
        frames = self._parser_rgb_rawframe(offset, frames_path, self.clip_length, step=1)

        mask_num = int(self.vq_res * self.vq_res * self.mask_ratio)
        mask_query_idx = np.zeros(self.vq_res * self.vq_res)
        sample_idx = np.array(random.sample(range(self.vq_res * self.vq_res), mask_num))
        mask_query_idx[sample_idx] = 1

        assert mask_query_idx.sum() == mask_num

        data = {
            'imgs': frames,
            'mask_query_idx': mask_query_idx,
            'modality': 'RGB',
            'num_clips': 1,
            'num_proposals':1,
            'clip_len': self.clip_length
        }

        return self.pipeline(data)