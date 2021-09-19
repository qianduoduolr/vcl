# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
from collections import *

import mmcv
from mmcv.runner import auto_fp16

from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.core.evaluation.metrics import JFM

import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):

        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 

class Encoder_M(nn.Module):
    def __init__(self, depth=50):
        super(Encoder_M, self).__init__()

        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        dic = {'type':'ResNet', 'depth':depth}
        resnet = build_backbone(dic)


        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_o):
        f = (in_f - self.mean) / self.std
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float() # add channel dim

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_o(o) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f
 
class Encoder_Q(nn.Module):
    def __init__(self, depth=50):
        super(Encoder_Q, self).__init__()
        dic = {'type':'ResNet', 'depth':depth}
        resnet = build_backbone(dic)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, mdim,scale_rate):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024//scale_rate, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512//scale_rate, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256//scale_rate, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)


    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        
        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p #, p2, p3, p4

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)

    def forward(self, x):
        x = self.atrous_conv(x)
        return F.relu(x,inplace=True)

class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()
        dilations = [1, 2, 4, 8]

        self.aspp1 = _ASPPModule(1024, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(1024, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(1024, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(1024, 256, 3, padding=dilations[3], dilation=dilations[3])
        self.conv1 = nn.Conv2d(1024, 256, 1, bias=False)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv1(x)
        return F.dropout(F.relu(x,inplace = True),p = 0.5,training=self.training)   

class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
 
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T*H*W) 
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb
 
        qi = q_in.view(B, D_e, H*W)  # b, emb, HW
 
        p = torch.bmm(mi, qi) # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1) # b, THW, HW

        mo = m_out.view(B, D_o, T*H*W) 
        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p

class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
 
    def forward(self, x, mode='all'):
        if mode == 'key':
            return self.Key(x)
        elif mode == 'value':
            return self.Value(x)
        else:
            return self.Key(x), self.Value(x)

@MODELS.register_module()
class STM(BaseModel):

    allowed_metrics = {'JFM':JFM}

    def __init__(self,
                 depth,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        """ Space-Time Memory Network for Video Object Segmentation

        Args:
            depth ([type]): ResNet depth for encoder
            pixel_loss ([type]): loss option
            train_cfg ([type], optional): [description]. Defaults to None.
            test_cfg ([type], optional): [description]. Defaults to None.
            pretrained ([type], optional): [description]. Defaults to None.
        """
        super(STM, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        scale_rate = (1 if (depth == 50 or depth == 101) else 4)
        
        # componets
        self.Encoder_M = Encoder_M(depth) 
        self.Encoder_Q = Encoder_Q(depth) 

        self.KV_M_r4 = KeyValue(1024//scale_rate, keydim=128//scale_rate, valdim=512//scale_rate)
        self.KV_Q_r4 = KeyValue(1024//scale_rate, keydim=128//scale_rate, valdim=512//scale_rate)

        self.Memory = Memory()
        self.Decoder = Decoder(256,scale_rate)

        # loss
        self.pixel_ce_loss = build_loss(pixel_loss)


        if test_cfg is not None:
            self.MEMORY_EVERY_FRAME = test_cfg.test_memory_every_frame
            self.MEMORY_NUM = test_cfg.memory_num

        self.freeze_bn()

    def freeze_bn(self):

        self.train()
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()

 
    def Pad_memory(self, mems, num_objects, K):
        pad_mems = []
        for mem in mems:
            pad_mem = ToCuda(torch.zeros(1, K, mem.size()[1], 1, mem.size()[2], mem.size()[3]))
            pad_mem[0,1:num_objects+1,:,0] = mem
            pad_mems.append(pad_mem)
        return pad_mems

    def memorize(self, frame, masks, num_objects): 
        # memorize a frame 
        num_objects = num_objects[0].item()
        _, K, H, W = masks.shape # B = 1

        (frame, masks), pad = pad_divide_by([frame, masks], 16, (frame.size()[2], frame.size()[3]))

        # make batch arg list
        B_list = {'f':[], 'm':[], 'o':[]}
        for o in range(1, num_objects+1): # 1 - no
            B_list['f'].append(frame)
            B_list['m'].append(masks[:,o])
            B_list['o'].append( (torch.sum(masks[:,1:o], dim=1) + \
                torch.sum(masks[:,o+1:num_objects+1], dim=1)).clamp(0,1) )

        # make Batch
        B_ = {}
        for arg in B_list.keys():
            B_[arg] = torch.cat(B_list[arg], dim=0)

        r4, _, _, _, _ = self.Encoder_M(B_['f'], B_['m'], B_['o'])
        k4, v4 = self.KV_M_r4(r4) # num_objects, 128 and 512, H/16, W/16
        k4, v4 = self.Pad_memory([k4, v4], num_objects=num_objects, K=K)
        return k4, v4

    def Soft_aggregation(self, ps, K):
        num_objects, H, W = ps.shape
        em = ToCuda(torch.zeros(1, K, H, W)) 
        em[0,0] =  torch.prod(1-ps, dim=0) # bg prob
        em[0,1:num_objects+1] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em)))
        return logit

    def segment(self, frame, keys, values, num_objects): 
        num_objects = num_objects[0].item()
        _, K, keydim, T, H, W = keys.shape # B = 1
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16
        
        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1) 
        r3e, r2e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1)
        
        # memory select kv:(1, K, C, T, H, W)
        m4, viz = self.Memory(keys[0,1:num_objects+1], values[0,1:num_objects+1], k4e, v4e)

        logits = self.Decoder(m4, r3e, r2e)
        ps = F.softmax(logits, dim=1)[:,1] # no, h, w  
        #ps = indipendant possibility to belong to each object
        
        logit = self.Soft_aggregation(ps, K) # 1, K, H, W

        if pad[2]+pad[3] > 0:
            logit = logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            logit = logit[:,:,:,pad[0]:-pad[1]]

        return logit


    def forward_train(self, Fs, Ms, num_objects):
        """
        Args:
            Fs ([type]): frames
            Ms ([type]): masks
            num_objects ([type]): number of objects

        Returns:
            [type]: loss dict
        """
        self.freeze_bn()

        losses = dict()

        Es = torch.zeros_like(Ms).cuda()
        Es[:,:,0] = Ms[:,:,0]

        n1_key, n1_value = self.memorize(Fs[:,:,0], Es[:,:,0], torch.tensor([num_objects]).cuda())

        # segment
        n2_logit = self.segment(Fs[:,:,1], n1_key, n1_value, torch.tensor([num_objects]).cuda())

        n2_label = torch.argmax(Ms[:,:,1],dim = 1).long().cuda()

        Es[:,:,1] = F.softmax(n2_logit, dim=1).detach()

        n2_key, n2_value = self.memorize(Fs[:,:,1], Es[:,:,1], torch.tensor([num_objects]).cuda())
        n12_keys = torch.cat([n1_key, n2_key], dim=3)
        n12_values = torch.cat([n1_value, n2_value], dim=3)

        # segment
        n3_logit = self.segment(Fs[:,:,2], n12_keys, n12_values, torch.tensor([num_objects]).cuda())
        n3_label = torch.argmax(Ms[:,:,2],dim = 1).long().cuda()

        losses['loss_pixel_ce_n2'] = self.pixel_ce_loss(n2_logit,n2_label)
        losses['loss_pixel_ce_n3'] = self.pixel_ce_loss(n3_logit, n3_label)

        return losses

    def forward_test(self, dataset, num_frames, num_objects, video,
                    save_image=False,
                    save_path=None,
                    iteration=None
                    ):
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert video is not None, (
                'evaluation with metrics must have gt images.')


        num_frames = num_frames.view(-1).item()
        num_objects = num_objects.view(-1).item()

        if self.MEMORY_EVERY_FRAME:
            to_memorize = [int(i) for i in np.arange(0, num_frames, step=self.MEMORY_EVERY_FRAME)]
        elif self.MEMORY_NUM:
            to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=self.MEMORY_NUM+2)[:-1]]
        else:
            raise NotImplementedError
        
        video_name = dataset.videos[video.view(-1).item()]

        F_last,M_last = dataset.load_single_image(video_name,0)
        F_last = F_last.unsqueeze(0).cuda()
        M_last = M_last.unsqueeze(0).cuda()
        E_last = M_last
        pred = np.zeros((num_frames,M_last.shape[3],M_last.shape[4]))
        all_Ms = []
        for t in range(1,num_frames):

            # memorize
            with torch.no_grad():
                prev_key, prev_value = self.memorize(F_last[:,:,0], E_last[:,:,0], torch.tensor([num_objects])) 

            if t-1 == 0: # 
                this_keys, this_values = prev_key, prev_value # only prev memory
            else:
                this_keys = torch.cat([keys, prev_key], dim=3)
                this_values = torch.cat([values, prev_value], dim=3)
            del prev_key,prev_value

            F_,M_ = dataset.load_single_image(video_name,t)

            F_ = F_.unsqueeze(0).cuda()
            M_ = M_.unsqueeze(0).cuda()
            all_Ms.append(M_.cpu().numpy())
            del M_

            # segment
            with torch.no_grad():
                logit = self.segment(F_[:,:,0], this_keys, this_values, torch.tensor([num_objects]))
            E = F.softmax(logit, dim=1)
            del logit
            # update
            if t-1 in to_memorize:
                keys, values = this_keys, this_values
                del this_keys,this_values
            pred[t] = torch.argmax(E[0], dim=0).cpu().numpy().astype(np.uint8)
            E_last = E.unsqueeze(2)
            F_last = F_
        Ms = np.concatenate(all_Ms,axis=2)

        all_res_masks = np.zeros((num_objects,pred.shape[0],pred.shape[1],pred.shape[2]))
        for i in range(1,num_objects+1):
            all_res_masks[i-1,:,:,:] = (pred == i).astype(np.uint8)
        all_res_masks = all_res_masks[:, 1:, :, :]
        all_gt_masks = Ms[0][1:1+num_objects]
        all_gt_masks = all_gt_masks[:, :, :, :]

        results = dict(eval_result=self.evaluate(all_gt_masks, all_res_masks, num_objects))

        return results
    

    def evaluate(self, all_gt_masks, all_res_masks, num_objects):

        eval_result = {}
        for metric in self.test_cfg.metrics:
            result_per_video = self.allowed_metrics[metric](all_gt_masks, all_res_masks, num_objects)
            eval_result[metric] = result_per_video

        return eval_result

    def forward(self, test_mode=False, **kwargs):
    
        if test_mode:
            return self.forward_test(**kwargs)

        return self.forward_train(**kwargs)

    def train_step(self, data_batch, optimizer):

        # parser loss
        losses = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(losses)

        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_vars.pop('loss')
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['Fs'])
        )

        return outputs