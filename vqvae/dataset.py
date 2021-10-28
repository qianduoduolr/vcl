import _init_paths
import os
import cv2
import lmdb
import random
import numpy as np

import torch
import torch.utils.data as data

import mmcv
import os.path as osp
import glob

from vcl.datasets.pipeline import RandomResizedCrop


class ImageFolderLMDB(data.Dataset):
    def __init__(self, lmdb_path, im_size):
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.im_size = im_size

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(str(index).encode())

        # load img
        img_np = np.frombuffer(byteflow, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img = img.astype(np.float32) / 255.

        # padding if image is too small
        h, w, _ = img.shape
        h_pad = max(0, self.im_size - h)
        w_pad = max(0, self.im_size - w)
        if h_pad != 0 or w_pad != 0:
            img = cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)

        # random crop
        h, w, _ = img.shape
        top = random.randint(0, h - self.im_size)
        left = random.randint(0, w - self.im_size)
        img = img[top:top + self.im_size, left:left + self.im_size, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))

        return img

    def __len__(self):
        return 1281167

class VideoFolderRGB(data.Dataset):
    def __init__(self, root,  
                       list_path, 
                       data_prefix, 
                       im_size=256
                       ):

        self.list_path = list_path
        self.root = root
        self.data_prefix = data_prefix

        self.load_annotations()

        self.randcrop = RandomResizedCrop(area_range=(0.6, 1.0))
        self.im_size = im_size

    def __len__(self):
        return len(self.samples)

    def load_annotations(self):
        
        self.samples = []
        self.mask_dir = osp.join(self.root, 'Annotations', '480p')
        self.video_dir = osp.join(self.root, 'JPEGImages', '480p')
        list_path = osp.join(self.list_path, f'davis{self.data_prefix}_{self.split}_list.txt')

        with open(list_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                sample = dict()
                vname, num_frames = line.strip('\n').split()
                sample['masks_path'] = sorted(glob.glob(osp.join(self.mask_dir, vname, '*.png')))
                sample['frames_path'] = sorted(glob.glob(osp.join(self.video_dir, vname, '*.jpg')))
                sample['video_path'] = osp.join(self.video_dir, vname)
                sample['num_frames'] = int(num_frames)
                self.samples.append(sample)

    def _parser_rgb_rawframe(self, offsets, frames_path, clip_length, step=1, flag='color', backend='cv2'):
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

    def __getitem__(self, index):

        sample = self.samples[index]
        num_frames = sample['num_frames']
        masks_path = sample['masks_path']
        frames_path = sample['frames_path']

        frame_idx = random.randint(0, num_frames-1)

        # load frame
        frame = self._parser_rgb_rawframe([frame_idx], frames_path, 1)
        
        # frame aug
        frame = self.randcrop(dict(imgs=frame))['imgs']
        frame = cv2.resize(frame, (self.im_size,self.im_size))

        frame = torch.from_numpy(frame.transpose(2, 0, 1))

        return frame

        


        
