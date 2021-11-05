import sys
sys.path.insert(0, '../raft/core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from tqdm import tqdm

DEVICE = 'cuda'
bound = 20

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def main(args):
    num_gpu = args.num_gpu

    # torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    model = torch.nn.parallel.DistributedDataParallel(RAFT(args), device_ids=[args.local_rank], broadcast_buffers=False)
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    vid_files = glob.glob(os.path.join(args.path,'*'))

    per_samples = len(vid_files) // num_gpu

    sub_files = vid_files[args.local_rank * per_samples:(args.local_rank+1) * per_samples] if args.local_rank != num_gpu -1 else \
        vid_files[args.local_rank * per_samples:]
    
    with torch.no_grad():
        for vid_file in tqdm(sub_files, total=len(sub_files)):
            images = glob.glob(os.path.join(vid_file, '*.png')) + \
                    glob.glob(os.path.join(vid_file, '*.jpg'))

            images = sorted(images)

            video_path = vid_file.replace('JPEGImages','Flows')
            os.makedirs(video_path, exist_ok=True)
            
            out_flows = []
            for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

                empty_img = 128 * np.ones((int(image1.shape[2]),int(image1.shape[3]),3)).astype(np.uint8)

                flow = np.clip(flow_up.permute(0,2,3,1).cpu().numpy(), -bound, bound)

                flow = (flow + bound) * (255.0 / (2*bound))
                flow = np.round(flow).astype('uint8')

                flow_img = empty_img.copy()
                flow_img[:,:,:2] = flow[:]

                dst_path = os.path.join(video_path, '{:05d}.jpg'.format(idx))
                cv2.imwrite(dst_path, flow_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='/gdata/lirui/models/optical_flow/raft-things.pth')
    parser.add_argument('--path', help="dataset for evaluation", default='/gdata/lirui/dataset/YouTube-VOS/2018/train/JPEGImages')
    parser.add_argument('--out', help="dataset for evaluation", default='/gdata/lirui/dataset/YouTube-VOS/2018/train/Flows')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--local_rank',  help='use small model', default=0)
    parser.add_argument('--num-gpu',  type=int, default=1, help='use small model')



    args = parser.parse_args()

    main(args)