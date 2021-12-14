from PIL import Image
import lmdb
import os
import cv2
import argparse
from tqdm import tqdm
import numpy as np
from glob import glob
from joblib import delayed, Parallel


target = 256

def create_lmdb_video_dataset_rgb(root_path, dst_path, workers=-1, quality=100, resize=True, save_jpg=False, save_lmdb=False):

    videos = glob(os.path.join(root_path,'*'))
    print('begin')

    def make_video(video_path, dst_path=None, resize=True, save_jpg=False):
        dst_file = video_path.replace('JPEGImages', f'LMDBImages_s{target}')
        dst_file_jpg = video_path.replace('JPEGImages', f'JPEGImages_s{target}')
        if save_lmdb:
            os.makedirs(dst_file, exist_ok=True)
        if save_jpg:
            os.makedirs(dst_file_jpg, exist_ok=True)

        frames = []
        idxs = []
        frame_names = sorted(glob(os.path.join(video_path, '*.jpg')))
        for frame_name in frame_names:
            frame = cv2.imread(frame_name)
            h, w, c = frame.shape

            if resize:
                if w >= h:
                    size = (int(target * w / h), int(target))
                else:
                    size = (int(target), int(target * h / w))

                frame = cv2.resize(frame, size, cv2.INTER_CUBIC)
            
            if save_jpg:
                file = os.path.join(dst_file_jpg, os.path.basename(frame_name))
                cv2.imwrite(file, frame)

            frames.append(frame)
            idxs.append(os.path.basename(frame_name))

        if save_lmdb:
            _, frame_byte = cv2.imencode('.jpg', frames[0],  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            env = lmdb.open(dst_file, frame_byte.nbytes * len(frames) * 50)
            frames_num = len(frames)
            for i in range(frames_num):
                txn = env.begin(write=True)
                key = 'image_{:05d}.jpg'.format(i+1)
                frame = frames[i]
                _, frame_byte = cv2.imencode('.jpg', frame,  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                txn.put(key.encode(), frame_byte)
                txn.commit()

            with open(os.path.join(dst_file, 'split.txt'),'a') as f:
                for idx in idxs:
                    f.write(idx + '\n')

    Parallel(n_jobs=workers)(delayed(make_video)(vp, dst_path, resize, save_jpg) for vp in tqdm(videos, total=len(videos)))



def create_lmdb_video_dataset_anno(root_path, dst_path, workers=-1, quality=100, resize=True, save_jpg=False, save_lmdb=False):

    videos = glob(os.path.join(root_path,'*'))
    print('begin')

    def make_video(video_path, dst_path=None, resize=True, save_jpg=False):
        dst_file = video_path.replace('Annotations', f'ANNOLMDB_s{target}')
        dst_file_jpg = video_path.replace('Annotations', f'Annotations_s{target}')
        if save_lmdb:
            os.makedirs(dst_file, exist_ok=True)
        if save_jpg:
            os.makedirs(dst_file_jpg, exist_ok=True)

        frames = []
        idxs = []
        frame_names = sorted(glob(os.path.join(video_path, '*.png')))
        for frame_name in frame_names:
            frame = Image.open(frame_name).convert('P')
            w, h = frame.size

            if resize:
                if w >= h:
                    size = (int(target * w / h), int(target))
                else:
                    size = (int(target), int(target * h / w))

                frame = frame.resize(size, Image.NEAREST)
            
            if save_jpg:
                file = os.path.join(dst_file_jpg, os.path.basename(frame_name))
                frame.save(file)

            frames.append(frame)
            idxs.append(os.path.basename(frame_name))

        if save_lmdb:
            raise NotImplemented

    Parallel(n_jobs=workers)(delayed(make_video)(vp, dst_path, resize, save_jpg) for vp in tqdm(videos, total=len(videos)))

def parse_option():
    parser = argparse.ArgumentParser('training')

    # dataset
    parser.add_argument('--root-path', type=str, default='/home/lr/dataset/YouTube-VOS/2018/train/Annotations', help='path of original data')
    parser.add_argument('--dst-path', type=str, default='LMDBImages_s256', help='path to store generated data')
    parser.add_argument('--num-workers', type=int, default=-1, help='num of workers to use')
    parser.add_argument('--resize', type=str, default='r', help='path to store generated data')
    parser.add_argument('--save-jpg', type=str, default='yes', help='path to store generated data')

    

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args =  parse_option()
    # create_lmdb_video_dataset_rgb(args.root_path, args.dst_path, workers=args.num_workers, resize=args.resize, save_jpg=args.save_jpg)
    create_lmdb_video_dataset_anno(args.root_path, args.dst_path, workers=args.num_workers, resize=args.resize, save_jpg=args.save_jpg)

