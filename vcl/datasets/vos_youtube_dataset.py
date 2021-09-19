import copy
import os.path as osp
from collections import defaultdict
from pathlib import Path
import glob
import os
import random
import torch
import numpy as np
import cv2
from PIL import Image

from mmcv import scandir

from .base_dataset import BaseDataset
from .registry import DATASETS
from .pipelines.aug import aug_heavy

MAX_OBJECT_NUM_PER_SAMPLE = 5

@DATASETS.register_module()
class VOS_youtube_dataset(BaseDataset):
    # for multi object, do shuffling
    def __init__(self, root, pipeline=None, test_mode=False):
        super().__init__(pipeline, test_mode)
        self.root = root

        self.K = 11
        self.skip = 0
        self.aug = aug_heavy()
        self.load_annotations()

    def load_annotations(self):
        self.mask_dir = os.path.join(self.root, 'Annotations')
        self.image_dir = os.path.join(self.root, 'JPEGImages')

        self.videos = [i.split('/')[-1] for i in glob.glob(os.path.join(self.image_dir,'*'))]
        self.num_frames = {}
        self.img_files = {}
        self.mask_files = {}
        for _video in self.videos:
            tmp_imgs = glob.glob(os.path.join(self.image_dir, _video,'*.jpg'))
            tmp_masks = glob.glob(os.path.join(self.mask_dir, _video,'*.png'))
            tmp_imgs.sort()
            tmp_masks.sort()
            self.img_files[_video] = tmp_imgs
            self.mask_files[_video] = tmp_masks
            self.num_frames[_video] = len(tmp_imgs)

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


    def prepare_train_data(self, index):
        video = self.videos[index]
        img_files = self.img_files[video]
        mask_files = self.mask_files[video]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        # info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((3,)+(384,384,)+(3,), dtype=np.float32)
        N_masks = np.empty((3,)+(384,384,), dtype=np.uint8)
        frames_ = []
        masks_ = []
        n1 = random.sample(range(0,self.num_frames[video] - 2),1)[0]
        n2 = random.sample(range(n1 + 1,min(self.num_frames[video] - 1,n1 + 2 + self.skip)),1)[0]
        n3 = random.sample(range(n2 + 1,min(self.num_frames[video],n2 + 2 + self.skip)),1)[0]
        frame_list = [n1,n2,n3]
        num_object = 0
        ob_list = []
        for f in range(3):
            img_file = img_files[frame_list[f]]
            tmp_frame = np.array(Image.open(img_file).convert('RGB'))
            try:
                mask_file = mask_files[frame_list[f]]  
                tmp_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                tmp_mask = 255

            h,w = tmp_mask.shape
            if h < w:
                tmp_frame = cv2.resize(tmp_frame, (int(w/h*480), 480), interpolation=cv2.INTER_LINEAR)
                tmp_mask = Image.fromarray(tmp_mask).resize((int(w/h*480), 480), resample=Image.NEAREST)  
            else:
                tmp_frame = cv2.resize(tmp_frame, (480, int(h/w*480)), interpolation=cv2.INTER_LINEAR)
                tmp_mask = Image.fromarray(tmp_mask).resize((480, int(h/w*480)), resample=Image.NEAREST) 

            frames_.append(tmp_frame)
            masks_.append(np.array(tmp_mask))

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