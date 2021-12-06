import copy
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
from .registry import DATASETS
from .pipelines.my_aug import ClipRandomSizedCrop, ClipMotionStatistic, ClipRandomHorizontalFlip
from .vos_davis_dataset import VOS_dataset_base

from .pipelines import Compose


@DATASETS.register_module()
class VOS_youtube_dataset_rgb(VOS_dataset_base):
    def __init__(self, root,  
                       list_path, 
                       data_prefix, 
                       clip_length=3,
                       num_clips=1,
                       pipeline=None, 
                       test_mode=False,
                       split='train',
                       year='2018',
                       load_to_ram=False
                       ):
        super().__init__(root, list_path, pipeline, test_mode, split)

        self.list_path = list_path
        self.root = root
        self.data_prefix = data_prefix
        self.split = split
        self.year = year
        self.clip_length = clip_length
        self.num_clips = num_clips

        self.load_annotations()

    def __len__(self):
        return len(self.samples)

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        self.mask_dir = osp.join(self.root, self.year, self.data_prefix['ANNO'])
        list_path = osp.join(self.list_path, f'youtube{self.year}_{self.split}_list.txt')


        with open(list_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                sample = dict()
                vname, num_frames = line.strip('\n').split()
                if int(num_frames) < self.clip_length: continue
                sample['frames_path'] = sorted(glob.glob(osp.join(self.video_dir, vname, '*.jpg')))
                sample['num_frames'] = int(num_frames)
                self.samples.append(sample)

    def prepare_train_data(self, idx):

        sample = self.samples[idx]
        frames_path = sample['frames_path']
        num_frames = sample['num_frames']

        offsets = [ random.randint(0, num_frames-self.clip_length) for i in range(self.num_clips) ]

        # load frame
        frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length)

        data = {
            'imgs': frames,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        }

        return self.pipeline(data)

@DATASETS.register_module()
class VOS_youtube_dataset_rgb_withbbox(VOS_youtube_dataset_rgb):

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        self.mask_dir = osp.join(self.root, self.year, self.data_prefix['ANNO'])
        self.meta = mmcv.load(osp.join(self.list_path, 'ytvos_s256_flow_raft.json'))[self.split]

        for vid in self.meta:
            sample = dict()
            vname = vid["base_path"].split('/')[-1]
            sample['frames_path'] = []
            sample['frames_bbox'] = []
            sample['num_frames'] = len(vid['frame'])
            if sample['num_frames'] < self.clip_length:
                continue
            for frame in vid['frame']:
                sample['frames_path'].append(osp.join(self.video_dir, vname, frame['img_path']))
                sample['frames_bbox'].append(frame['objs'][0]['bbox'])

            self.samples.append(sample)
    
    def prepare_train_data(self, idx):

        sample = self.samples[idx]

        frames_path = sample['frames_path']
        frames_bbox = sample['frames_bbox']
        num_frames = sample['num_frames']

        offsets = [ random.randint(0, num_frames-self.clip_length) for i in range(self.num_clips) ]

        # load frame
        frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length)
        bboxs = []
        for offset in offsets:
            for i in range(self.clip_length):
                bboxs.append(frames_bbox[offset+i])

        data = {
            'imgs': frames,
            'bboxs': bboxs,
            'mask_sample_size': (32, 32),
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        }

        return self.pipeline(data)
    



@DATASETS.register_module()
class VOS_youtube_dataset_mlm(VOS_dataset_base):
    def __init__(self, root,  
                       list_path, 
                       data_prefix, 
                       mask_ratio=0.15,
                       clip_length=3,
                       num_clips=1,
                       vq_size=32,
                       pipeline=None, 
                       test_mode=False,
                       split='train',
                       year='2018',
                       load_to_ram=False
                       ):
        super().__init__(root, list_path, pipeline, test_mode, split)


        self.list_path = list_path
        self.root = root
        self.data_prefix = data_prefix
        self.year = year
        self.load_to_ram = load_to_ram

        self.clip_length = clip_length
        self.num_clips = num_clips
        self.vq_res = vq_size
        self.mask_ratio = mask_ratio

        self.load_annotations()

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        self.mask_dir = osp.join(self.root, self.year, self.data_prefix['ANNO'])
        list_path = osp.join(self.list_path, f'youtube{self.year}_{self.split}_list.txt')

        if self.test_mode:
            self.test_dic = {}
            with open(osp.join(self.list_path, 'test_records.txt'), 'r') as f:
                for line in f.readlines():
                    name, offset, s_idx = line.strip('\n').split()
                    self.test_dic[name] = [int(offset), int(s_idx)]

        with open(list_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                sample = dict()
                vname, num_frames = line.strip('\n').split()
                sample['frames_path'] = sorted(glob.glob(osp.join(self.video_dir, vname, '*.jpg')))
                sample['masks_path'] = sorted(glob.glob(osp.join(self.mask_dir, vname, '*.png')))
                sample['num_frames'] = int(num_frames)
                if sample['num_frames'] < self.clip_length:
                    continue
                if self.load_to_ram:
                    sample['frames'] = self._parser_rgb_rawframe([0], sample['frames_path'], sample['num_frames'], step=1)
                self.samples.append(sample)
    
    def prepare_test_data(self, idx):
        sample = self.samples[idx]
        frames_path = sample['frames_path']
        masks_path = sample['masks_path']
        num_frames = sample['num_frames']
        vid_name = frames_path[0].split('/')[-2]

        offset = [self.test_dic[vid_name][0]]
        for i in range(self.num_clips - 1):
            offset.append(offset[-1] + 1)

        if self.load_to_ram:
            frames = (sample['frames'])[offset[0]:offset[0]+self.clip_length]
        else:
            frames = self._parser_rgb_rawframe(offset, frames_path, self.clip_length, step=1)
        sample_idx = np.array([self.test_dic[vid_name][1]])

        assert sample_idx.shape[0] == 1

        data = {
            'imgs': frames,
            'mask_query_idx': sample_idx,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        }

        return self.pipeline(data)

    def prepare_train_data(self, idx):
        
        sample = self.samples[idx]
        frames_path = sample['frames_path']
        num_frames = sample['num_frames']

        offset = [ random.randint(0, num_frames-self.clip_length) for i in range(self.num_clips)]


        if self.load_to_ram:
            frames = (sample['frames'])[offset[0]:offset[0]+self.clip_length]
        else:
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
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        }

        return self.pipeline(data)


@DATASETS.register_module()
class VOS_youtube_dataset_mlm_motion(VOS_youtube_dataset_mlm):
    def __init__(self, size, p=0.7, flow_context=3, crop_ratio=0.6, **kwargs):
        self.flow_context = flow_context
        super().__init__(**kwargs)
        self.trans = transforms.Compose([ClipRandomSizedCrop(size=size, scale=(crop_ratio, 1)),
                                        ClipRandomHorizontalFlip(p=0.5)]
                                        )
        self.motion_statistic = ClipMotionStatistic(input_size=size, mag_size=self.vq_res)
        self.p = p


    def load_annotations(self):
        self.samples = []
        
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        self.mask_dir = osp.join(self.root, self.year, self.data_prefix['ANNO'])
        self.flows_dir = osp.join(self.root, self.year, self.data_prefix['FLOW'])
        list_path = osp.join(self.list_path, f'youtube{self.year}_{self.split}_list.txt')

        with open(list_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                sample = dict()
                vname, num_frames = line.strip('\n').split()
                sample['frames_path'] = sorted(glob.glob(osp.join(self.video_dir, vname, '*.jpg')))[:-1]
                sample['masks_path'] = sorted(glob.glob(osp.join(self.mask_dir, vname, '*.png')))[:-1]
                sample['num_frames'] = int(num_frames) - 1
                sample['flows_path'] = []
                flows_path_all = sorted(glob.glob(osp.join(self.flows_dir, vname, '*.jpg')))
                if sample['num_frames'] < self.clip_length:
                    continue
                if self.load_to_ram:
                    sample['frames'] = self._parser_rgb_rawframe([0], sample['frames_path'], sample['num_frames'], step=1)
                    sample['flows'] = self._parser_rgb_rawframe([0], flows_path_all, sample['num_frames'], step=1)


                if self.flow_context is not -1:
                    for frame_path in sample['frames_path']:
                        flow_path = frame_path.replace(self.data_prefix['RGB'], self.data_prefix['FLOW'])
                        try:
                            index = flows_path_all.index(flow_path)
                            flow_context = flows_path_all[max(index-1,0):index+2] if not self.load_to_ram else [max(index-1,0), index+2]
                            sample['flows_path'].append(flow_context)
                        except Exception as e:
                            index = None
                            break
                    if index == None:
                        continue
                else:
                    for frame_path in sample['frames_path']:
                        flow_path = frame_path.replace(self.data_prefix['RGB'], self.data_prefix['FLOW'])
                        sample['flows_path'].append([flow_path])

                self.samples.append(sample)


    def prepare_train_data(self, idx):
        sample = self.samples[idx]
        frames_path = sample['frames_path']
        flows_path = sample['flows_path']
        num_frames = sample['num_frames']

        offset = [ random.randint(0, num_frames-self.clip_length) for i in range(self.num_clips)]


        if self.load_to_ram:
            frames = (sample['frames'])[offset[0]:offset[0]+self.clip_length]
            flows = (sample['flows'])[flows_path[offset[0]][0]:flows_path[offset[0]][1]]
        else:
            frames = self._parser_rgb_rawframe(offset, frames_path, self.clip_length, step=1)
            sample_flows_path = flows_path[offset[0]+1]
            flows = self._parser_rgb_rawframe([0], sample_flows_path, len(sample_flows_path), step=1)

        results = dict(imgs=frames, flows=flows, with_flow=True)
        results = self.trans(results)
        mag = self.motion_statistic(results['flows'])

        mask_num = int(self.vq_res * self.vq_res * self.mask_ratio)
        mask_query_idx = np.zeros(self.vq_res * self.vq_res)
        if random.random() <= self.p:
            mag = mag.reshape(-1)
            idxs = np.argsort(mag)[::-1][:mask_num]
            mask_query_idx[idxs] = 1
        else:
            sample_idx = np.array(random.sample(range(self.vq_res * self.vq_res), mask_num))
            mask_query_idx[sample_idx] = 1

        assert mask_query_idx.sum() == mask_num

        data = {
            'imgs': results['imgs'],
            'mask_query_idx': mask_query_idx,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        }

        return self.pipeline(data)


@DATASETS.register_module()
class VOS_youtube_dataset_mlm_withbbox(VOS_youtube_dataset_mlm):
    def __init__(self, size, p=1.0, crop_ratio=0.6, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        self.mask_dir = osp.join(self.root, self.year, self.data_prefix['ANNO'])
        self.meta = mmcv.load(osp.join(self.list_path, 'ytvos_s256_flow_raft.json'))[self.split]

        for vid in self.meta:
            sample = dict()
            vname = vid["base_path"].split('/')[-1]
            sample['frames_path'] = []
            sample['frames_bbox'] = []
            sample['num_frames'] = len(vid['frame'])
            if sample['num_frames'] < self.clip_length:
                continue
            for frame in vid['frame']:
                sample['frames_path'].append(osp.join(self.video_dir, vname, frame['img_path']))
                sample['frames_bbox'].append(frame['objs'][0]['bbox'])

            self.samples.append(sample)
    
    def prepare_train_data(self, idx):
        sample = self.samples[idx]
        frames_path = sample['frames_path']
        frames_bbox = sample['frames_bbox']
        num_frames = sample['num_frames']

        offset = [ random.randint(0, num_frames-self.clip_length) for i in range(self.num_clips)]


        if self.load_to_ram:
            frames = (sample['frames'])[offset[0]:offset[0]+self.clip_length]
        else:
            frames = self._parser_rgb_rawframe(offset, frames_path, self.clip_length, step=1)
        
        bboxs = [ frames_bbox[offset[0] + i]  for i in range(self.clip_length * self.num_clips)]

        data = {
            'imgs': frames,
            'bboxs': bboxs,
            'mask_ratio': self.mask_ratio,
            'mask_sample_size': (self.vq_res, self.vq_res),
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        }

        return self.pipeline(data)


@DATASETS.register_module()
class VOS_youtube_dataset_mlm_withbbox_random(VOS_youtube_dataset_mlm):
    def __init__(self, size, p=1.0, crop_ratio=0.6, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        self.mask_dir = osp.join(self.root, self.year, self.data_prefix['ANNO'])
        self.meta = mmcv.load(osp.join(self.list_path, 'ytvos_s256_flow_raft.json'))[self.split]

        for vid in self.meta:
            sample = dict()
            vname = vid["base_path"].split('/')[-1]
            sample['frames_path'] = []
            sample['frames_bbox'] = []
            sample['num_frames'] = len(vid['frame'])
            if sample['num_frames'] < self.clip_length:
                continue
            for frame in vid['frame']:
                sample['frames_path'].append(osp.join(self.video_dir, vname, frame['img_path']))
                sample['frames_bbox'].append(frame['objs'][0]['bbox'])

            self.samples.append(sample)
    
    def prepare_train_data(self, idx):
        sample = self.samples[idx]
        frames_path = sample['frames_path']
        frames_bbox = sample['frames_bbox']
        num_frames = sample['num_frames']

        offset = [ random.randint(0, num_frames-self.clip_length) for i in range(self.num_clips)]

        if self.load_to_ram:
            frames = (sample['frames'])[offset[0]:offset[0]+self.clip_length]
        else:
            frames = self._parser_rgb_rawframe(offset, frames_path, self.clip_length, step=1)
        
        bboxs = [ frames_bbox[offset[0] + i]  for i in range(self.clip_length * self.num_clips)]

        if random.random() <= self.p:
            data = {
                'imgs': frames,
                'bboxs': bboxs,
                'mask_ratio': self.mask_ratio,
                'mask_sample_size': (self.vq_res, self.vq_res),
                'modality': 'RGB',
                'num_clips': self.num_clips,
                'num_proposals':1,
                'clip_len': self.clip_length
            }
            return self.pipeline(data)

        else:
            mask_num = int(self.vq_res * self.vq_res * self.mask_ratio)
            mask_query_idx = np.zeros(self.vq_res * self.vq_res).astype(np.uint8)
            sample_idx = np.array(random.sample(range(self.vq_res * self.vq_res), mask_num))
            mask_query_idx[sample_idx] = 1

            data = {
                'imgs': frames,
                'mask_query_idx': mask_query_idx,
                'modality': 'RGB',
                'num_clips': self.num_clips,
                'num_proposals':1,
                'clip_len': self.clip_length
            }

            return self.pipeline(data)