import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data
import random
import glob
import pdb
import cv2
from dataset.aug import aug_heavy
import json
import copy

MAX_OBJECT_NUM_PER_SAMPLE = 5
class Youtube_MO_Train(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations')
        self.image_dir = os.path.join(root, 'JPEGImages')

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

        self.K = 11
        self.skip = 0
        self.aug = aug_heavy()



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


    def __getitem__(self, index):
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
        return Fs, Ms, num_objects, info

class Youtube_MO_Train_Contrast(Youtube_MO_Train):
    
    def __init__(self, list_path, **kwargs):
        super(Youtube_MO_Train_Contrast, self).__init__(**kwargs)
        self.list_path = list_path
        self.frame_meta_data = self.load_annotations()

        for _video in self.videos:
            tmp_imgs = glob.glob(os.path.join(self.image_dir, _video,'*.jpg'))
            tmp_masks = glob.glob(os.path.join(self.mask_dir, _video,'*.png'))
            tmp_imgs.sort()
            tmp_masks.sort()

            imgs = []
            masks = []

            for tmp_img, tmp_mask in zip(tmp_imgs, tmp_masks):
                frame_id = tmp_img.split('/')[-1][:-4]
                if len(self.frame_meta_data[_video][frame_id]) > 0:
                    imgs.append(tmp_img)
                    masks.append(tmp_mask)

            self.img_files[_video] = imgs
            self.mask_files[_video] = masks
            self.num_frames[_video] = len(imgs)

    def load_annotations(self):
        with open(self.list_path, 'r') as f:
            meta_data = json.load(f)
        return meta_data

    def __getitem__(self, index):

        video = self.videos[index]
        img_files = self.img_files[video]
        mask_files = self.mask_files[video]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        video_frame_info = self.frame_meta_data[video]
        # info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((4,)+(384,384,)+(3,), dtype=np.float32)
        N_masks = np.empty((3,)+(384,384,), dtype=np.uint8)
        frames_ = []
        masks_ = []
        frame_obj_masks_ = []
        n1 = random.sample(range(0,self.num_frames[video] - 2),1)[0]
        n2 = random.sample(range(n1 + 1,min(self.num_frames[video] - 1,n1 + 2 + self.skip)),1)[0]
        n3 = random.sample(range(n2 + 1,min(self.num_frames[video],n2 + 2 + self.skip)),1)[0]
        frame_list = [n1,n2,n3]

        frame_infos = list([ video_frame_info[img_files[i].split('/')[-1][:-4]] for i in frame_list ])

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
                tmp_mask_ = Image.fromarray(tmp_mask).resize((int(w/h*480), 480), resample=Image.NEAREST)  
            else:
                tmp_frame = cv2.resize(tmp_frame, (480, int(h/w*480)), interpolation=cv2.INTER_LINEAR)
                tmp_mask_ = Image.fromarray(tmp_mask).resize((480, int(h/w*480)), resample=Image.NEAREST)
            
            
            frames_.append(tmp_frame)
            masks_.append(np.array(tmp_mask_))

        frames_aug ,masks_aug = self.aug(frames_,masks_)

        for m in masks_aug:
            frame_obj_masks_.append(cv2.resize(m, (24,24), interpolation=cv2.INTER_NEAREST).astype(np.uint8))

        for f in range(3):
            masks_aug[f],num_object,ob_list = self.mask_process(masks_aug[f],f,num_object,ob_list)
            N_frames[f],N_masks[f] = frames_aug[f],masks_aug[f]

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
        Ms_obj, labels = self.obj_mask_process(frame_obj_masks_, frame_infos, video)
        Ms_obj = torch.from_numpy(Ms_obj)

        if num_object == 0:
            num_object += 1
        num_objects = torch.LongTensor([num_object])

        return Fs, Ms, Ms_obj, labels, num_objects, info

    def obj_mask_process(self, frame_obj_masks, frame_infos, name):

        obj_candidate = []
        obj_labels ={x[1]:x[0] for x in frame_infos[-1]}

        query_mask = frame_obj_masks[-1]
        for obj in frame_infos[-1]:
            i = obj[1]
            if (query_mask == i).sum() > 0:
                obj_candidate.append(i)
        
        if len(obj_candidate) == 0:
            masks = np.zeros((3,24,24)).astype(np.uint8)
            rand_loc = [random.randint(0,23), random.randint(0,23)]
            masks[2, rand_loc[0], rand_loc[1]] = 1

            return masks, 0

        else:

            query_obj_num = random.sample(obj_candidate, 1)[0]

            label = obj_labels[query_obj_num]
            masks = []

            pixel_num_sum = 0
            for i,frame_obj_mask in enumerate(frame_obj_masks[:-1]):
                frame_info = frame_infos[i]
                for j, obj_info in enumerate(frame_info):
                    if obj_info[1] == query_obj_num:
                        obj_mask = (frame_obj_mask ==  query_obj_num).astype(np.uint8)
                        break
                    elif j == len(frame_info) - 1:
                        obj_mask = np.zeros((24,24)).astype(np.uint8)

                pixel_num_sum += obj_mask.sum()
                masks.append(obj_mask)
            
            masks.append((query_mask == query_obj_num).astype(np.uint8))

            if masks[-1].sum() <= 0:
                print(name)

            return np.array(masks), label
            


if __name__ == '__main__':
    from helpers import overlay_davis
    import matplotlib.pyplot as plt
    import os
    import pdb


    dataset = Youtube_MO_Train('/smart/haochen/cvpr/data/YOUTUBE-VOS/train/')
    dataset.skip = 10
    palette = Image.open('/smart/haochen/cvpr/data/DAVIS/Annotations/480p/blackswan/00000.png').getpalette()

    output_dir = 'tmp'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i,(Fs,Ms,num_objects,info) in enumerate(dataset):
        pred = np.argmax(Ms.numpy(), axis=0).astype(np.uint8)
        img_list = []
        for f in range(3):
            pF = (Fs[:,f].permute(1,2,0).numpy()*255.).astype(np.uint8)
            pE = pred[f]
            canvas = overlay_davis(pF, pE, palette)
            img = np.concatenate([pF,canvas],axis = 0)
            img_list.append(img)
        out_img = np.concatenate(img_list,axis = 1)
        out_img = Image.fromarray(out_img)
        out_img.save(os.path.join(output_dir, str(i).zfill(5) + '.jpg'))
        pdb.set_trace()