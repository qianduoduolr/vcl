from __future__ import division
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
from utils.logger import setup_logger
from utils.util import *
from utils.lr_scheduler import *
from contrast.NCEContrast import *

# general libs
import cv2
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import random
import json


### My libs
from dataset.dataset import DAVIS_MO_Test
from dataset.davis import DAVIS_MO_Train
from dataset.youtube import Youtube_MO_Train, Youtube_MO_Train_Contrast
from model.model import STM, STM_SSL
from eval import evaluate
from utils.helpers import overlay_davis

def get_arguments():
	parser = argparse.ArgumentParser(description="SST")
	parser.add_argument("--Ddavis", type=str, help="path to data",default='/home/lr/dataset/DAVIS/')
	parser.add_argument("--Dyoutube", type=str, help="path to youtube-vos",default='/home/lr/dataset/YouTube-VOS/')
	parser.add_argument("--list-path", type=str, help="path to youtube-vos",default='/home/lr/dataset/YouTube-VOS/train/generated_frame_wise_meta.json')

	parser.add_argument("--batch-size", type=int, help="batch size",default=1)
	parser.add_argument("--num-workers", type=int, help="batch size",default=4)
	parser.add_argument("--base-learning-rate", type=float, help="batch size",default=0.01)
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
	parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
	parser.add_argument('--warmup-epoch', type=int, default=0, help='warmup epoch')
	parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
	parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")


	parser.add_argument("--max_skip", type=int, help="max skip between training frames",default=25)
	parser.add_argument("--change_skip_step", type=int, help="change max skip per x iter",default=3000)
	parser.add_argument("--total-iter", type=int, help="total iter num",default=200000)
	parser.add_argument("--eval-freq", type=int, help="evaluate per x iters",default=100000)
	parser.add_argument("--print-freq", type=int, help="log per x iters",default=1000)
	parser.add_argument("--save-freq", type=int, help="log per x iters",default=40000)

	parser.add_argument("--davis-freq", type=int, help="log per x iters",default=13)
	parser.add_argument("--pretrained-model",type=str,default='/home/lr/models/segmentation/coco_pretrained_resnet50_679999_169.pth')
	parser.add_argument("--output-dir",type=str,default='./output')
	parser.add_argument("--sample_rate",type=float,default=0.08)
	parser.add_argument("--nce-t",type=float,default=0.2)
	parser.add_argument("--nce-k",type=int,default=65536)


	parser.add_argument("--backbone", type=str, help="backbone ['resnet50', 'resnet18']",default='resnet18')

	# dist
	parser.add_argument("--multi", type=str2bool, default='false')
	parser.add_argument("--expname", type=str, default='ssl', help="path to data")
	parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel', default=0)

	args = parser.parse_args()
	return args

def main(args, logger):
	rate = args.sample_rate

	DATA_ROOT = args.Ddavis
	palette = Image.open(DATA_ROOT + '/Annotations/480p/blackswan/00000.png').getpalette()
	torch.backends.cudnn.benchmark = True


	# for Youtube
	YOUTUBE_ROOT = args.Dyoutube
	Trainset1 = Youtube_MO_Train_Contrast(list_path=args.list_path, root='{}train/'.format(YOUTUBE_ROOT))

	if args.multi:
		train_sampler = torch.utils.data.distributed.DistributedSampler(Trainset1)
		Trainloader1 = torch.utils.data.DataLoader(
                Trainset1, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,sampler=train_sampler,
                drop_last=True)
	else:
		Trainloader1 = data.DataLoader(Trainset1, batch_size=1, num_workers=1,shuffle = True, pin_memory=True)
	loader_iter1 = iter(Trainloader1)

	Testloader = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(17,'val'), single_object=False)
	logger.info('Build dataset successfully')

	# build model
	model = STM_SSL(args.backbone).cuda()
	contrast = MemorySTM(128, args.nce_k, args.nce_t).cuda()

	if args.multi:
		model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)
		if args.pretrained_model:
			logger.info('Loading weights:{}'.format(args.pretrained_model))
			state_dict = torch.load(args.pretrained_model,map_location=torch.device('cpu'))
			[un, miss] = model.load_state_dict(state_dict, strict=False)
			logger.info(un)
			logger.info(miss)

		if dist.get_rank() == 0:
			summary_writer = SummaryWriter(log_dir=os.path.join(args.output_dir,'logs'))
		else:
			summary_writer = None
	logger.info('Build model successfully')


	model.train()
	for module in model.modules():
		if isinstance(module, torch.nn.modules.BatchNorm1d):
			module.eval()
		if isinstance(module, torch.nn.modules.BatchNorm2d):
			module.eval()
		if isinstance(module, torch.nn.modules.BatchNorm3d):
			module.eval()
	
	ct_loss = NCESoftmaxLoss().cuda()

	# optimizer
	params = list([v for k,v in model.named_parameters() if k.find('Encoder_Q_M') == -1 and k.find('proj_m') == -1])
	optimizer = torch.optim.SGD(params,
                                lr=args.base_learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

	scheduler = get_scheduler(optimizer, len(Trainloader1), args)
	change_skip_step = args.change_skip_step
	max_skip = 25
	skip_n = 0
	max_jf = 0
	ratio = args.total_iter / 800000
	sampler_epoch = 0

	loss_meter_cts = AverageMeter()
	time_meter = AverageMeter()

	for iter_ in range(args.total_iter):
		start = time.time()

		if (iter_ + 1) % (change_skip_step * ratio) == 0:
			if skip_n < max_skip:
				skip_n += 1
			Trainset1.change_skip(skip_n//5)
			loader_iter1 = iter(Trainloader1)
		try:
			Fs, Ms, Ms_obj, labels, num_objects, info = next(loader_iter1)
		except:
			if args.multi:
				sampler_epoch += 1
				Trainloader1.sampler.set_epoch(sampler_epoch)
			loader_iter1 = iter(Trainloader1)
			Fs, Ms, Ms_obj, labels, num_objects, info = next(loader_iter1)


		seq_name = info['name'][0]
		num_frames = info['num_frames'][0].item()
		num_frames = 3
		Fs = Fs.cuda()
		Ms = Ms.cuda()

		Ms_obj = Ms_obj.cuda()
		labels = labels.cuda()

		Es = torch.zeros_like(Ms).cuda()
		Es[:,:,0] = Ms[:,:,0]

		q_feat = model(Fs[:,:,2], mode='query')
		q_feat = nn.functional.normalize(q_feat, p=2, dim=1)


		mask3 = Ms_obj[:,2,:,:].unsqueeze(1)
		num_pixel3 = mask3.sum().item()

		mask2 = Ms_obj[:,1,:,:].unsqueeze(1)
		num_pixel2 = mask2.sum().item()

		mask1 = Ms_obj[:,0,:,:].unsqueeze(1)
		num_pixel1 = mask1.sum().item()

		num_pixel = num_pixel1 + num_pixel2

		feat_dim = q_feat.size(1)


		if num_pixel == 0:
			# num_pixel3 == 1 or more
			with torch.no_grad():
				k3_feat = model(Fs[:,:,2], mode='key')
				k3_feat = nn.functional.normalize(k3_feat, p=2, dim=1)
			if num_pixel3 == 1:
				q_mask = mask3.repeat(1,feat_dim,1,1).to(torch.bool)
				q_feat = (q_feat * q_mask).reshape(1, feat_dim, -1).mean(-1)
				q_region = q_feat
				total_k_feat = q_feat.unsqueeze(1).repeat(1,16,1).reshape(16, -1)

			else:
				q_mask = mask3.repeat(1,feat_dim,1,1).to(torch.bool)
				q_feats = torch.masked_select(q_feat, q_mask).reshape(1, feat_dim, num_pixel3).permute(0,2,1)
				q_feat = q_feats[:, random.randint(0, num_pixel3 -1), :]
				q_region = q_feats.mean(1)

				pixel_k_feat = torch.masked_select(k3_feat, q_mask).reshape(1, feat_dim, num_pixel3).permute(0,2,1)
				region_k_feat = pixel_k_feat.mean(1)
				if num_pixel3 < 15:
					ratio = math.ceil(15 / num_pixel3)
					pixel_k_feat = pixel_k_feat.unsqueeze(1).repeat(1, ratio, 1, 1).reshape(1, -1, feat_dim)[:,:15,:]
				else:
					indexs = random.sample(range(num_pixel3), 15)
					pixel_k_feat =  pixel_k_feat[:,indexs]
				total_k_feat = torch.cat([pixel_k_feat, region_k_feat.unsqueeze(1)], 1).reshape(16, -1)

			labels = labels.repeat(16)
			
		else:
			with torch.no_grad():
				k1_feat = model(Fs[:,:,0], mode='key')
				k2_feat = model(Fs[:,:,1], mode='key')
				k1_feat = nn.functional.normalize(k1_feat, p=2, dim=1)
				k2_feat = nn.functional.normalize(k2_feat, p=2, dim=1)
			
			# select query pixel and region
			q_mask = mask3.repeat(1,feat_dim,1,1).to(torch.bool)
			q_feats = torch.masked_select(q_feat, q_mask).reshape(1, feat_dim, num_pixel3).permute(0,2,1)
			q_feat = q_feats[:, random.randint(0, num_pixel3-1), :] if num_pixel3 > 1 else q_feats.squeeze(1)
			q_region = q_feats.mean(1)

			# select key region and pixel
			mask1 = mask1.repeat(1,feat_dim,1,1).to(torch.bool)
			mask2 = mask2.repeat(1,feat_dim,1,1).to(torch.bool)
			if num_pixel1 > 0 and num_pixel2 > 0:
				pixel_k_feat1 = torch.masked_select(k1_feat, mask1).reshape(1, feat_dim, num_pixel1).permute(0,2,1)
				pixel_k_feat2 = torch.masked_select(k2_feat, mask2).reshape(1, feat_dim, num_pixel2).permute(0,2,1)
				pixel_k_feat = torch.cat([pixel_k_feat1, pixel_k_feat2], 1)
				region_k_feat = torch.stack([pixel_k_feat1.mean(1),pixel_k_feat2.mean(1)],1)
			elif num_pixel1 > 0:
				pixel_k_feat = torch.masked_select(k1_feat, mask1).reshape(1, feat_dim, num_pixel1).permute(0,2,1)
				region_k_feat = pixel_k_feat.mean(1, keepdim=True)
			else:
				pixel_k_feat = torch.masked_select(k2_feat, mask2).reshape(1, feat_dim, num_pixel2).permute(0,2,1)
				region_k_feat = pixel_k_feat.mean(1, keepdim=True)

			region_k = region_k_feat.size(1)

			if num_pixel < 16 - region_k:
				ratio = math.ceil((16 - region_k) / pixel_k_feat.size(1))
				pixel_k_feat = pixel_k_feat.unsqueeze(1).repeat(1, ratio, 1, 1).reshape(1, -1, feat_dim)[:,:16 - region_k,:]
			else:
				indexs = random.sample(range(num_pixel), 16 - region_k)
				pixel_k_feat =  pixel_k_feat[:,indexs]

			total_k_feat = torch.cat([region_k_feat, pixel_k_feat], 1).reshape(16, -1)
			labels = labels.repeat(16)

		assert total_k_feat.size(1) == 128
		assert q_feat.size(1) == 128

		total_k_feat_all = dist_collect(total_k_feat) if args.multi else total_k_feat
		labels_all = dist_collect(labels) if args.multi else labels.reshape(16)

		out, out_region, m, l_pos = contrast(q_feat, q_region, total_k_feat, labels, total_k_feat_all, labels_all)

		# scl loss
		loss_cts = - torch.log( (F.softmax(out, dim=1) * m ).sum(1)).mean()
		loss_cts_region = - torch.log( (F.softmax(out_region, dim=1) * m ).sum(1) ).mean()
		# print(loss_cts, loss_cts_region, labels[0], m.sum() /16)
		# loss_cts = ct_loss(out)
		# loss_cts_region = ct_loss(out)
		loss = loss_cts + loss_cts_region

		loss.backward()

		loss_meter_cts.update(loss_cts.item()+loss_cts_region.item())

		optimizer.step()
		scheduler.step()

		optimizer.zero_grad()

		batch_time = time.time() - start
		time_meter.update(batch_time)

		if args.multi:
			moment_update(model.module.Encoder_Q, model.module.Encoder_Q_M, 0.999)
			moment_update(model.module.proj,model.module.proj_m, 0.999)
		else:
			moment_update(model.Encoder_Q, model.Encoder_Q_M, 0.999)
			moment_update(model.proj,model.proj_m, 0.999)

		if args.multi:
			if dist.get_rank() == 0:
				summary_writer.add_scalar('Train/loss_cts', loss_cts.item(), iter_ + 1)
				summary_writer.add_scalar('Train/loss_cts_region', loss_cts_region.item(), iter_ + 1)


		if (iter_+1) % args.print_freq == 0:
			# print('iteration:{}, loss:{}, remaining iteration:{}'.format(iter_,loss_momentum/log_iter, args.total_iter - iter_))
			logger.info('Train: [{:>3d}]/[{:>4d}] BT={:>0.3f} / {:>0.3f} Loss={:>0.3f} {:>0.3f} {:>0.3f}  / {:>0.3f} '.format(iter_+1, args.total_iter, batch_time, time_meter.avg, loss.item(), loss_cts.item(), loss_cts_region.item(), loss_meter_cts.avg))
			loss_meter_cts.reset()
		
		if (iter_+1) % args.save_freq == 0:
			if dist.get_rank() == 0:
				output_dir = os.path.join(args.output_dir, 'checkpoints')
				if not os.path.exists(output_dir):
					os.makedirs(output_dir, exist_ok=True)
				torch.save(model.state_dict(), os.path.join(output_dir,'davis_youtube_{}_{}.pth'.format(args.backbone,str(iter_))))


if __name__ == '__main__':
	opt = get_arguments()
	file_path = os.path.dirname(os.path.abspath(__file__))
	if opt.multi:
		file = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
		exp_path = 'log-sslstm-{args.backbone}-iters{args.total_iter}-nce-t{args.nce_t}-nce-k{args.nce_k}-expname{args.expname}'.format(args=opt)
		opt.output_dir = os.path.join(opt.output_dir, exp_path + file)
		torch.cuda.set_device(opt.local_rank)

		torch.distributed.init_process_group(backend='nccl', init_method='env://')
		cudnn.benchmark = True
		logs_dir = os.path.join(opt.output_dir, 'logs')
		os.makedirs(logs_dir, exist_ok=True)
		distributed_rank = dist.get_rank() if opt.multi else 0
		logger = setup_logger(output=logs_dir, distributed_rank=distributed_rank, name="sslstm")

		if dist.get_rank() == 0:
			path = os.path.join(logs_dir, "config.json")
			with open(path, 'w') as f:
				json.dump(vars(opt), f, indent=2)
			logger.info("Full config saved to {}".format(path))
			os.system('cp -r {} {}'.format(file_path, opt.output_dir))
	else:
		logger = setup_logger(output=None, distributed_rank=0, name="sslstm")

	main(opt, logger)