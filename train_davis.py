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
from dataset.youtube import Youtube_MO_Train
from model.model import STM
from eval import evaluate
from utils.helpers import overlay_davis

def get_arguments():
	parser = argparse.ArgumentParser(description="SST")
	parser.add_argument("--Ddavis", type=str, help="path to data",default='/home/lr/dataset/DAVIS/')
	parser.add_argument("--Dyoutube", type=str, help="path to youtube-vos",default='/home/lr/dataset/YouTube-VOS/')
	parser.add_argument("--batch-size", type=int, help="batch size",default=1)
	parser.add_argument("--num-workers", type=int, help="batch size",default=4)

	parser.add_argument("--max_skip", type=int, help="max skip between training frames",default=25)
	parser.add_argument("--change_skip_step", type=int, help="change max skip per x iter",default=3000)
	parser.add_argument("--total-iter", type=int, help="total iter num",default=800000)
	parser.add_argument("--eval-freq", type=int, help="evaluate per x iters",default=400000)
	parser.add_argument("--print-freq", type=int, help="log per x iters",default=100)
	parser.add_argument("--pretrained-model",type=str,default='')
	parser.add_argument("--output-dir",type=str,default='./output')
	parser.add_argument("--sample_rate",type=float,default=0.08)
	parser.add_argument("--backbone", type=str, help="backbone ['resnet50', 'resnet18']",default='resnet18')

	# dist
	parser.add_argument("--multi", type=str2bool, default='false')
	parser.add_argument("--expname", type=str, default='', help="path to data")
	parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

	
	args = parser.parse_args()
	return args

def main(args, logger):
	logger.info('Main start')
	rate = args.sample_rate

	DATA_ROOT = args.Ddavis
	palette = Image.open(DATA_ROOT + '/Annotations/480p/blackswan/00000.png').getpalette()
	torch.backends.cudnn.benchmark = True

	print(args.local_rank)
	logger.info('Build dataset')

	# for DAVIS
	Trainset = DAVIS_MO_Train(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(17,'train'), single_object=False)
	if args.multi:
		train_sampler = torch.utils.data.distributed.DistributedSampler(Trainset)
		Trainloader = torch.utils.data.DataLoader(
                Trainset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,sampler=train_sampler,
                drop_last=True)
	else:
		Trainloader = data.DataLoader(Trainset, batch_size=1, num_workers=1,shuffle = True, pin_memory=True)
	loader_iter = iter(Trainloader)

	print(args.local_rank)
	# for Youtube
	YOUTUBE_ROOT = args.Dyoutube
	Trainset1 = Youtube_MO_Train('{}train/'.format(YOUTUBE_ROOT))

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
	print(args.local_rank)
	logger.info('Build dataset successfully')

	# build model
	model = STM(args.backbone, logger, opt).cuda()
	print('hh')
	logger.info('Build model')

	if args.pretrained_model:
		logger.info('Loading weights:{}'.format(args.pretrained_model))
		state_dict = torch.load(args.pretrained_model)
		state_dict_new = {k.replace('module.',''):v for k,v in state_dict.items()}
		model.load_state_dict(state_dict_new)

	if args.multi:
		model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
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
	
	criterion = nn.CrossEntropyLoss()
	criterion.cuda()
	optimizer = torch.optim.Adam(model.parameters(),lr = 1e-5,eps=1e-8, betas=[0.9,0.999])

	def adjust_learning_rate(iteration,power = 0.9):
		lr = 1e-5 * pow((1 - 1.0 * iteration / args.total_iter), power)
		return lr

	loss_momentum = 0
	change_skip_step = args.change_skip_step
	max_skip = 25
	skip_n = 0
	max_jf = 0
	ratio = args.total_iter / 800000
	sampler_epoch = 0

	for iter_ in range(args.total_iter):
    	
		start = time.time()

		if (iter_ + 1) % (1000 * ratio) == 0:
			lr = adjust_learning_rate(iter_)
			for param_group in optimizer.param_groups:
				param_group["lr"] = lr

		if (iter_ + 1) % (change_skip_step * ratio) == 0:
			if skip_n < max_skip:
				skip_n += 1
			Trainset1.change_skip(skip_n//5)
			loader_iter1 = iter(Trainloader1)
			Trainset.change_skip(skip_n)
			loader_iter = iter(Trainloader)

		if random.random() < rate:
			try:
				Fs, Ms, num_objects, info = next(loader_iter)
			except:
				if args.multi:
					sampler_epoch += 1
					Trainloader.sampler.set_epoch(sampler_epoch)
				loader_iter = iter(Trainloader)
				Fs, Ms, num_objects, info = next(loader_iter)
		else:
			try:
				Fs, Ms, num_objects, info = next(loader_iter1)
			except:
				if args.multi:
					sampler_epoch += 1
					Trainloader1.sampler.set_epoch(sampler_epoch)
				loader_iter1 = iter(Trainloader1)
				Fs, Ms, num_objects, info = next(loader_iter1)
		
		seq_name = info['name'][0]
		num_frames = info['num_frames'][0].item()
		num_frames = 3
		Fs = Fs.cuda()
		Ms = Ms.cuda()

		Es = torch.zeros_like(Ms).cuda()
		Es[:,:,0] = Ms[:,:,0]

		n1_key, n1_value = model(Fs[:,:,0], Es[:,:,0], torch.tensor([num_objects]).cuda())
		n2_logit = model(Fs[:,:,1], n1_key, n1_value, torch.tensor([num_objects]).cuda())

		n2_label = torch.argmax(Ms[:,:,1],dim = 1).long().cuda()
		n2_loss = criterion(n2_logit,n2_label)

		Es[:,:,1] = F.softmax(n2_logit, dim=1).detach()

		n2_key, n2_value = model(Fs[:,:,1], Es[:,:,1], torch.tensor([num_objects]).cuda())
		n12_keys = torch.cat([n1_key, n2_key], dim=3)
		n12_values = torch.cat([n1_value, n2_value], dim=3)
		n3_logit = model(Fs[:,:,2], n12_keys, n12_values, torch.tensor([num_objects]).cuda())


		n3_label = torch.argmax(Ms[:,:,2],dim = 1).long().cuda()
		n3_loss = criterion(n3_logit,n3_label)

		Es[:,:,2] = F.softmax(n3_logit, dim=1)

		loss = n2_loss + n3_loss

		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

		batch_time = time.time() - start

		if args.multi:
			if dist.get_rank() == 0:
				summary_writer.add_scalar('Train/loss', loss.item(), iter_ + 1)

		if (iter_+1) % args.print_freq == 0:
			# print('iteration:{}, loss:{}, remaining iteration:{}'.format(iter_,loss_momentum/log_iter, args.total_iter - iter_))
			logger.info('Train: [{:>3d}]/[{:>4d}] BT={:>0.3f} Loss={:>0.3f} '.format(iter_+1, args.total_iter, batch_time, loss.item()))


		if (iter_+1) % args.eval_freq == 0:
			if args.multi:
				if dist.get_rank() == 0:
					output_dir = os.path.join(args.output_dir, 'checkpoints')
					if not os.path.exists(output_dir):
						os.makedirs(output_dir)
					torch.save(model.state_dict(), os.path.join(output_dir,'davis_youtube_{}_{}.pth'.format(args.backbone,str(iter_))))
					
					model.eval()
					
					logger.info('Evaluate at iter: ' + str(iter_))
					g_res = evaluate(model,Testloader,['J','F'])

					logger.info('J&F-Mean: {}, J-Mean:{}, J-Recall: {}, J-Decay:{}, F-Mean:{}, F-Recall:{}, F-Decay{}'.format(g_res[0], g_res[1], g_res[2], g_res[3], g_res[4], g_res[5], g_res[6]))
					
					model.train()
					for module in model.modules():
						if isinstance(module, torch.nn.modules.BatchNorm1d):
							module.eval()
						if isinstance(module, torch.nn.modules.BatchNorm2d):
							module.eval()
						if isinstance(module, torch.nn.modules.BatchNorm3d):
							module.eval()
	
if __name__ == '__main__':
	opt = get_arguments()
	file_path = os.path.dirname(os.path.abspath(__file__))
	print(opt.local_rank)
	if opt.multi:
		file = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
		exp_path = 'log-stm-{args.backbone}-iters{args.total_iter}-expname{args.expname}'.format(args=opt)
		opt.output_dir = os.path.join(opt.output_dir, exp_path + file)
		torch.cuda.set_device(opt.local_rank)

		torch.distributed.init_process_group(backend='nccl', init_method='env://')
		cudnn.benchmark = True
		logs_dir = os.path.join(opt.output_dir, 'logs')
		os.makedirs(logs_dir, exist_ok=True)

		print(dist.get_rank())
		distributed_rank = dist.get_rank() if opt.multi else 0
		logger = setup_logger(output=logs_dir, distributed_rank=distributed_rank, name="stm")

		if dist.get_rank() == 0:
			path = os.path.join(logs_dir, "config.json")
			with open(path, 'w') as f:
				json.dump(vars(opt), f, indent=2)
			logger.info("Full config saved to {}".format(path))
			os.system('cp -r {} {}'.format(file_path, opt.output_dir))
	else:
		logger = setup_logger(output=None, distributed_rank=0, name="stm")

	main(opt, logger)