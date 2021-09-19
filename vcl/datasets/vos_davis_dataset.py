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
from .pipelines.aug import aug_heavy


MAX_OBJECT_NUM_PER_SAMPLE = 5

@DATASETS.register_module()
class VOS_davis_dataset(BaseDataset):

    def __init__(self, imset, resolution, single_object, root, pipeline=None, test_mode=False):
        super().__init__(pipeline, test_mode)

        self.K = 11

        self.root = root
        self.single_object = single_object
        self.imset = imset
        self.resolution = resolution

        self.aug = aug_heavy()
        self.load_annotations()

    def load_annotations(self):

        self.mask_dir = os.path.join(self.root, 'Annotations', self.resolution)
        self.mask480_dir = os.path.join(self.root, 'Annotations', '480p')
        self.image_dir = os.path.join(self.root, 'JPEGImages', self.resolution)
        _imset_dir = os.path.join(self.root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, self.imset)

        self.skip = 0
        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)


    def __len__(self):
        return len(self.videos)

    def change_skip(self,f):
        self.skip = f

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

    def load_single_image(self, video, f):  
    
        N_frames = np.empty((1,)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((1,)+self.shape[video], dtype=np.uint8)

        img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))

        N_frames[0] = np.array(Image.open(img_file).convert('RGB'))/255.
        try:
            mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
            N_masks[0] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        except:
            N_masks[0] = 255
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms


    def prepare_train_data(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((3,)+(384,384,)+(3,), dtype=np.float32)
        N_masks = np.empty((3,)+(384,384,), dtype=np.uint8)
        frames_ = []
        masks_ = []
        n1 = random.sample(range(0,self.num_frames[video] - 2),1)[0]
        n2 = random.sample(range(n1+1,min(self.num_frames[video] - 1,n1 + 2 + self.skip)),1)[0]
        n3 = random.sample(range(n2+1,min(self.num_frames[video],n2 + 2 + self.skip)),1)[0]
        frame_list = [n1,n2,n3]
        num_object = 0
        ob_list = []

        for f in range(3):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(frame_list[f]))
            tmp_frame = np.array(Image.open(img_file).convert('RGB'))
            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(frame_list[f]))  
                tmp_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                tmp_mask = 255

            frames_.append(tmp_frame.astype(np.float32))
            masks_.append(tmp_mask)
        frames_,masks_ = self.aug(frames_,masks_)

        for f in range(3):
            masks_[f],num_object,ob_list = self.mask_process(masks_[f],f,num_object,ob_list)
            N_frames[f],N_masks[f] = frames_[f],masks_[f]
        
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()

        if num_object == 0:
            num_object += 1
        num_objects = torch.LongTensor([num_object])

        data = {
            'Fs': Fs,
            'Ms': Ms,
            'num_objects':num_objects
        }

        return data

    def prepare_test_data(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        num_objects = torch.LongTensor([int(self.num_objects[video])])

        data = {
            'num_objects': num_objects,
            'num_frames': info['num_frames'],
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

