import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2

import torch
import torchvision
from torch.utils import data

import random
import glob
from collections import *
import pdb

from .base_dataset import BaseDataset
from .registry import DATASETS
from .pipelines.my_aug import aug_heavy


MAX_OBJECT_NUM_PER_SAMPLE = 5

@DATASETS.register_module()
class VOS_dataset_base(BaseDataset):

    def __init__(self, root,  
                       list_path, 
                       pipeline=None, 
                       test_mode=False
                       ):
        super().__init__(pipeline, test_mode)

        self.root = root
        self.list_path = list_path


    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def mask_process(self,mask,f,num_object,ob_list):
        n = num_object
        mask_ = np.zeros(mask.shape).astype(np.uint8)
        if f == 0:
            for i in range(1,11):
                if np.sum(mask == i) > 0:
                    n += 1
                    ob_list.append(i)
            if n > MAX_OBJECT_NUM_PER_SAMPLE:
                n = MAX_OBJECT_NUM_PER_SAMPLE
                ob_list = random.sample(ob_list,n)
        for i,l in enumerate(ob_list):
            mask_[mask == l] = i + 1
        return mask_,n,ob_list

    def _parser_rgb_jpg_cv(self, offsets, frames_path):
        """read frame"""
        frame_list_all = []
        for idx, offset in enumerate(offsets):
            frame_path = frames_path[offset]
            frame = cv2.imread(frame_path)
            frame_list_all.append(frame)
        return frame_list_all
    
    def _parser_mask(self, offsets, masks_path):
        """read mask"""
        mask_list_all = []
        for idx, offset in enumerate(offsets):
            mask_path = masks_path[offset]
            mask = np.array(Image.open(mask_path).convert('P'), dtype=np.uint8)
            mask_list_all.append(mask)
        return mask_list_all


@DATASETS.register_module()
class VOS_davis_dataset_stm(VOS_dataset_base):

    def __init__(self, root,  
                       list_path, 
                       resolution, 
                       year, 
                       single_object, 
                       sample_type='stm', 
                       pipeline=None, 
                       test_mode=False
                       ):
        super().__init__(root, list_path, pipeline, test_mode)

        self.sample_type = sample_type
        self.aug = aug_heavy()
        self.K = 11
        self.skip = 0
        
        self.resolution = resolution
        self.year = year

    def load_annotations(self):
        
        self.samples = []

        mask_dir = osp.join(self.root, 'Annotations', self.resolution)
        video_dir = osp.join(self.root, 'JPEGImages', self.resolution)
        list_path = osp.join(self.list_path, f'davis{self.year}_train_list.txt') if not self.test_mode else  osp.join(self.list_path, f'davis{self.year}_val_list.txt')

        with open(list_path, 'r') as f:
            for line in f.readlines():
                sample = dict()
                vname, num_frames = line.strip('\n').split()
                sample['frames_path'] = sorted(glob.glob(osp.join(video_dir, vname, '*.jpg')))
                sample['masks_path'] = sorted(glob.glob(osp.join(mask_dir, vname, '*.png')))
                sample['num_frames'] = int(num_frames)

                _mask = np.array(Image.open(sample['masks_path'][0]).convert('P'), dtype=np.uint8)
                sample['num_objects'] = np.max(_mask)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

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

    def prepare_test_data(self, index):
        sample = self.samples[index]
        num_frames = sample['num_frames']
        num_objects = sample['num_objects']

        num_objects = torch.LongTensor([int(num_objects)])

        data = {
            'num_objects': num_objects,
            'num_frames': num_frames,
            'video': index
        }

        return data


    def evaluate(self, results, logger=None):

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_results = defaultdict(list)  # a dict of list

        for res in results:
            for metric, sub_metric in res.items():
                for metric_, v in sub_metric.items():
                    eval_results[metric_].extend(v)
        
         # average the results
        eval_results = {
            metric: np.mean(values)
            for metric, values in eval_results.items()
        }
        return eval_results
    
    def load_single_image(self, sample, f):  

        frame_file = sample['frames_path'][f]

        N_frames = (np.array(Image.open(frame_file).convert('RGB'))/255)[None, :]

        mask_file = sample['masks_path'][f]
        N_masks = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)[None, :]

        Fs = torch.from_numpy(np.transpose(N_frames, (3, 0, 1, 2))).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks)).float()
            num_objects = torch.LongTensor([int(sample['num_objects'])])
            return Fs, Ms