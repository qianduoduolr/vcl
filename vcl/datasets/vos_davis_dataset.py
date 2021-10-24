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
from mmcv.utils import print_log
import mmcv
from vcl.utils import add_prefix, terminal_is_available

from .davis2017.evaluation import DAVISEvaluation


from .base_dataset import BaseDataset
from .registry import DATASETS
from .pipelines.my_aug import aug_heavy


MAX_OBJECT_NUM_PER_SAMPLE = 5

@DATASETS.register_module()
class VOS_dataset_base(BaseDataset):
    def __init__(self, root,  
                       list_path, 
                       pipeline=None, 
                       test_mode=False,
                       filename_tmpl='{:05d}.jpg',
                       split='train'
                       ):
        super().__init__(pipeline, test_mode)

        self.list_path = list_path
        self.root = root
        self.filename_tmpl = filename_tmpl
        self.split = split

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


@DATASETS.register_module()
class VOS_davis_dataset_test(VOS_dataset_base):
    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [191, 0, 0], [64, 128, 0], [191, 128, 0], [64, 0, 128],
               [191, 0, 128], [64, 128, 128], [191, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 191, 0], [128, 191, 0], [0, 64, 128],
               [128, 64, 128]]
    
    def __init__(self, root,  
                       list_path, 
                       data_prefix, 
                       pipeline=None, 
                       test_mode=False,
                       task='semi-supervised',
                       split='val'
                       ):
        super().__init__(root, list_path, pipeline, test_mode=test_mode, split=split)

        self.task = task

        self.list_path = list_path
        self.root = root
        self.data_prefix = data_prefix

        self.load_annotations()

    def load_annotations(self):
        
        self.samples = []
        self.mask_dir = osp.join(self.root, 'Annotations', '480p')
        self.video_dir = osp.join(self.root, 'JPEGImages', '480p')
        list_path = osp.join(self.list_path, f'davis{self.data_prefix}_{self.split}_list.txt')

        with open(list_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                if idx >= 1: break
                sample = dict()
                vname, num_frames = line.strip('\n').split()
                sample['masks_path'] = sorted(glob.glob(osp.join(self.mask_dir, vname, '*.png')))
                sample['frames_path'] = sorted(glob.glob(osp.join(self.video_dir, vname, '*.jpg')))
                sample['video_path'] = osp.join(self.video_dir, vname)
                sample['num_frames'] = int(num_frames)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def prepare_test_data(self, index):
        sample = self.samples[index]
        num_frames = sample['num_frames']
        masks_path = sample['masks_path']
        frames_path = sample['frames_path']
        
        # load frames and masks
        frames = self._parser_rgb_rawframe([0], frames_path, num_frames)
        ref = np.array(self._parser_rgb_rawframe([0], masks_path, 1, flag='unchanged', backend='pillow')[0])
        original_shape = frames[0].shape[:2]

        data = {
            'imgs': frames,
            'ref_seg_map': ref,
            'video_path': sample['video_path'],
            'original_shape': original_shape,
            'modality': 'RGB',
            'num_clips': 1,
            'clip_len': num_frames
        }

        return self.pipeline(data)

    def davis_evaluate(self, results, output_dir, logger=None):
        dataset_eval = DAVISEvaluation(
            davis_root=self.root, task=self.task, gt_set='val')
        if isinstance(results, str):
            metrics_res = dataset_eval.evaluate(results)
        else:
            assert len(results) == len(self)
            for vid_idx in range(len(self)):
                assert len(results[vid_idx]) == \
                       self.samples[vid_idx]['num_frames'] or \
                       isinstance(results[vid_idx], str)
            if output_dir is None:
                tmp_dir = tempfile.TemporaryDirectory()
                output_dir = tmp_dir.name
            else:
                tmp_dir = None
                mmcv.mkdir_or_exist(output_dir)

            if terminal_is_available():
                prog_bar = mmcv.ProgressBar(len(self))
            for vid_idx in range(len(results)):
                cur_results = results[vid_idx]
                if isinstance(cur_results, str):
                    file_path = cur_results
                    cur_results = np.load(file_path)
                    os.remove(file_path)
                for img_idx in range(
                        self.samples[vid_idx]['num_frames']):
                    result = cur_results[img_idx].astype(np.uint8)
                    img = Image.fromarray(result)
                    img.putpalette(
                        np.asarray(self.PALETTE, dtype=np.uint8).ravel())
                    video_path = self.samples[vid_idx]['video_path']
                    raw_img = Image.open(osp.join(video_path, self.filename_tmpl.format(img_idx)))

                    # blend
                    img = img.convert('RGBA')
                    raw_img = raw_img.convert('RGBA')
                    blend_image = Image.blend(raw_img, img, 0.3)

                    save_path = osp.join(
                    output_dir, osp.relpath(video_path, self.video_dir), 'blend',
                    self.filename_tmpl.format(img_idx).replace('.jpg', '_blend.png'))

                    save_path_mask = osp.join(
                    output_dir, osp.relpath(video_path, self.video_dir),
                    self.filename_tmpl.format(img_idx).replace(
                        'jpg', 'png'))

                    mmcv.mkdir_or_exist(osp.dirname(save_path))
                    mmcv.mkdir_or_exist(osp.dirname(save_path_mask))

                    blend_image.save(save_path)
                    img.save(save_path_mask)

                if terminal_is_available():
                    prog_bar.update()
            metrics_res = dataset_eval.evaluate(output_dir)
            if tmp_dir is not None:
                tmp_dir.cleanup()

        J, F = metrics_res['J'], metrics_res['F']

        # Generate dataframe for the general results
        g_measures = [
            'J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall',
            'F-Decay'
        ]
        final_mean = (np.mean(J['M']) + np.mean(F['M'])) / 2.
        g_res = np.array([
            final_mean,
            np.mean(J['M']),
            np.mean(J['R']),
            np.mean(J['D']),
            np.mean(F['M']),
            np.mean(F['R']),
            np.mean(F['D'])
        ])
        g_res = np.reshape(g_res, [1, len(g_res)])
        print_log(f'\nGlobal results for {self.split}', logger=logger)
        table_g = pd.DataFrame(data=g_res, columns=g_measures)
        print_log('\n' + table_g.to_string(index=False), logger=logger)

        # Generate a dataframe for the per sequence results
        seq_names = list(J['M_per_object'].keys())
        seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
        J_per_object = [J['M_per_object'][x] for x in seq_names]
        F_per_object = [F['M_per_object'][x] for x in seq_names]
        table_seq = pd.DataFrame(
            data=list(zip(seq_names, J_per_object, F_per_object)),
            columns=seq_measures)
        print_log(f'\nPer sequence results for  {self.split}', logger=logger)
        print_log('\n' + table_seq.to_string(index=False), logger=logger)

        eval_results = table_g.to_dict('records')[0]

        return eval_results

    def evaluate(self, results, metrics='daivs', output_dir=None, logger=None):
        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['davis']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        eval_results = dict()
        if mmcv.is_seq_of(results, np.ndarray) and results[0].ndim == 4:
            num_feats = results[0].shape[0]
            for feat_idx in range(num_feats):
                cur_results = [result[feat_idx] for result in results]
                eval_results.update(
                    add_prefix(
                        self.davis_evaluate(cur_results, output_dir, logger),
                        prefix=f'feat_{feat_idx}'))
        elif mmcv.is_seq_of(results, list):
            num_feats = len(results[0])
            for feat_idx in range(num_feats):
                cur_results = [result[feat_idx] for result in results]
                eval_results.update(
                    add_prefix(
                        self.davis_evaluate(cur_results, output_dir, logger),
                        prefix=f'feat_{feat_idx}'))
        else:
            eval_results.update(
                self.davis_evaluate(results, output_dir, logger))
        copypaste = []
        for k, v in eval_results.items():
            if 'J&F' in k:
                copypaste.append(f'{float(v)*100:.2f}')
        print_log(f'Results copypaste  {",".join(copypaste)}', logger=logger)
        return eval_results



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