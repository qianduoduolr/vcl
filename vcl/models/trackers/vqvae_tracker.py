# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
from collections import *
from tkinter.tix import Tree

import mmcv
from mmcv.runner import auto_fp16, load_checkpoint
from vcl.models.common.correlation import frame_transform

from vcl.models.losses.losses import Ce_Loss
from vcl.models.common.correlation import *
from vcl.models.common.utils import  *

from ..base import BaseModel
from ..builder import build_backbone, build_loss, build_components, build_model
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.utils import *




import torch.nn as nn
import torch
import torch.nn.functional as F
import math


@MODELS.register_module()
class Vqvae_Tracker(BaseModel):

    def __init__(self,
                 backbone,
                 vqvae,
                 patch_size,
                 pretrained_vq,
                 post_convolution=None,
                 vq_size=32,
                 temperature=1.0,
                 sim_siam_head=None,
                 multi_head_weight=[1.0],
                 ce_loss=None,
                 mse_loss=None,
                 fc=True,
                 train_cfg=None,
                 test_cfg=None,
                 per_ref=True,
                 mask_radius=-1,
                 temp_window=False,
                 scaling_att=False,
                 norm=False,
                 pretrained=None
                 ):
        """ original vqvae tracker
        """
        super(Vqvae_Tracker, self).__init__()
        if not isinstance(vqvae, list):
            vqvae = list([vqvae])
            pretrained_vq = list([pretrained_vq])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.patch_size = patch_size
        self.num_head = len(vqvae)
        self.multi_head_weight = multi_head_weight
        self.per_ref = per_ref
        self.temperature = temperature
        self.temp_window = temp_window
        self.norm = norm
        self.scaling_att = scaling_att
        
        self.logger = get_root_logger()

        self.backbone = build_backbone(backbone)
        if post_convolution is not None:
            self.post_convolution = nn.Conv2d(post_convolution['in_c'], post_convolution['out_c'], post_convolution['ks'], 1, post_convolution['pad'])
        else:
            self.post_convolution = None
            
            
        if sim_siam_head is not None:
            self.head = build_components(sim_siam_head)
            self.head.init_weights()
        else:
            self.head = None

        self.vq_type = vqvae[0].type
        if self.vq_type != 'DALLE_Encoder':
            for idx, i in enumerate(range(len(vqvae))):
                i = str(i).replace('0', '')
                setattr(self, f'vqvae{i}', build_model(vqvae[idx]).cuda())
                _ = load_checkpoint(getattr(self, f'vqvae{i}'), pretrained_vq[idx], map_location='cpu')
                self.logger.info(f'load {i}th pretrained VQVAE successfully')
                setattr(self, f'vq_emb{i}', getattr(self, f'vqvae{i}').quantize.embed)
                setattr(self, f'n_embed{i}', vqvae[idx].n_embed)
                setattr(self, f'vq_t{i}', temperature)
                setattr(self, f'vq_enc{i}', getattr(self, f'vqvae{i}').encode)
        else:
            raise NotImplementedError

        # loss
        self.ce_loss = build_loss(ce_loss) if ce_loss else None
        self.mse_loss = build_loss(mse_loss) if mse_loss else None

        # corr
        if patch_size != -1:
            from spatial_correlation_sampler import SpatialCorrelationSampler
            self.correlation_sampler = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=patch_size,
                stride=1,
                padding=0,
                dilation=1)

        # fc
        self.fc = fc
        if self.fc:
            for i in range(len(vqvae)):
                i = str(i).replace('0', '')
                setattr(self, f'predictor{i}', nn.Linear(self.backbone.feat_dim, getattr(self, f'n_embed{i}')))
        else:
            self.embedding_layer = nn.Linear(self.backbone.feat_dim, self.vq_emb.shape[0])

        # init weights
        self.init_weights(pretrained)
        
        if mask_radius != -1:
            self.mask = make_mask(vq_size, mask_radius)
        else:
            self.mask = None
        
            
    def init_weights(self, pretrained):
        
        self.backbone.init_weights()
        if pretrained:
            _ = load_checkpoint(self, pretrained, map_location='cpu')
            self.logger.info('load pretrained model successfully')
        
        return 

    def forward_train(self, imgs, mask_query_idx, jitter_imgs=None):

        bsz, num_clips, t, c, h, w = imgs.shape
    
        # vqvae tokenize for query frame
        with torch.no_grad():
            out_ind= []
            out_quant = []
            for i in range(self.num_head):
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc = getattr(self, f'vq_enc{i}')
                vqvae.eval()
                emb, quant, _, ind, _ = vq_enc(imgs[:, 0, -1])
                
                if self.per_ref:
                    ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                    quant = quant.unsqueeze(1).repeat(1, t-1, 1, 1, 1).permute(0,1,3,4,2).flatten(0,3).detach()
                else:
                    ind = ind.reshape(-1, 1).long().detach()
                    quant = quant.permute(0,2,3,1).flatten(0,2).detach()
                    
                out_ind.append(ind)
                out_quant.append(quant)
                
        mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1) if self.per_ref else mask_query_idx.bool()
            
        if jitter_imgs is not None:
            imgs = jitter_imgs

        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            out, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            out, att = non_local_attention(tar, refs, per_ref=self.per_ref, temprature=self.temperature)

        losses = {}

        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                if self.fc:
                    predict = getattr(self, f'predictor{i}')(out)
                else:
                    predict = self.embedding_layer(out)
                    predict = nn.functional.normalize(predict, dim=-1)
                    vq_emb = getattr(self, f'vq_emb{i}')
                    predict = torch.mm(predict, nn.functional.normalize(vq_emb, dim=0))
                    # predict = torch.div(predict, 0.1) # temperature is set to 0.1、
                    
                loss = self.ce_loss(predict, out_ind[idx])
                losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
                
        
        if self.mask:
            mask = torch.ones(*att.shape).cuda() - self.mask
            target = torch.zeros(*att.shape).cuda()
            loss = F.l1_loss(mask*att, target, reduction='none')
            losses['att_sparse_loss'] = 10 * (loss * mask_query_idx.unsqueeze(-1)).sum() / mask_query_idx.sum()
                    
        return losses

    def forward_test_correspondence(self, imgs, mask_query_idx,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        bsz, num_clips, t, c, h, w = imgs.shape
        
        tar = self.backbone(imgs[:,0, 0])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(1,t)])
        atts = []
        
        # for short term
        out, att_s = non_local_attention(tar, [refs[0]], per_ref=self.per_ref)
        
        # for long term
        if True:
            _, att = non_local_attention(tar, refs, per_ref=self.per_ref)
            att = att.reshape(bsz, att_s.shape[-1], t-1, -1)
            for i in range(t-1):
                atts.append(att[:,:,i])
        else:
            att_l = torch.eye(att_s.shape[-1]).repeat(bsz, 1, 1).cuda()
            for i in range(t-1):
                if i == 0:
                    att_l = torch.einsum('bij,bjk->bik', [att_l, att_s]) 
                else:
                    _, att = non_local_attention(refs[i-1], [refs[i]], per_ref=self.per_ref)
                    att_l = torch.einsum('bij,bjk->bik', [att_l, att]) 
                    
                atts.append(att_l)
        
        if len(atts) == 0:
            atts.append(att_s)
            
        # visualize_att(imgs, atts, iteration, False, mask_query_idx, tar.shape[-1], self.patch_size, dst_path=save_path, norm_mode='mean-std')

        # vqvae tokenize for query frame
        with torch.no_grad():
            _, quant, diff, ind, embed = self.vq_enc(imgs[:, 0, -1])
            ind = ind.reshape(-1, 1).long().detach()


        if self.backbone.out_indices[0] == 3:
            if self.fc:
                predict = self.predictor(out)
            else:
                if out.shape[-1] != self.vq_emb.shape[0]:
                    predict = self.embedding_layer(out)
                else:
                    predict = out
                predict = nn.functional.normalize(predict, dim=-1)
                predict = torch.mm(predict, nn.functional.normalize(self.vq_emb, dim=0))
                predict = torch.div(predict, self.vq_t)

            out2 = torch.argmax(predict.cpu().detach(), axis=1).numpy()
            out1 = ind.cpu().numpy()

            return out1, out2
        else:
            return None
        
    
    

@MODELS.register_module()
class Vqvae_Tracker_V5(Vqvae_Tracker):

    def __init__(self,
                 l1_loss=None,
                 **kwargs
                 ):
        """ ce_loss with cross video/sparstity constrant
        """
        super(Vqvae_Tracker_V5, self).__init__(**kwargs)
        
        self.l1_loss = build_loss(l1_loss)

    def forward_train(self, imgs, mask_query_idx, jitter_imgs=None):
        bsz, num_clips, t, c, h, w = imgs.shape

        # vqvae tokenize for query frame
        with torch.no_grad():
            out_ind= []
            for i in range(self.num_head):
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc = getattr(self, f'vq_enc{i}')
                vqvae.eval()
                _, _, _, ind, _ = vq_enc(imgs[:, 0, -1])
                ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                out_ind.append(ind)

        if jitter_imgs is not None:
            imgs = jitter_imgs

        mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)

        fs = self.backbone(imgs.flatten(0,2))
        fs = fs.reshape(bsz, t, *fs.shape[-3:])
        tar, refs = fs[:, -1], fs[:, :-1]
        
        out, att = non_local_attention(tar, refs, per_ref=True)
        out_cross_video, att_cross_video = inter_intra_attention(tar, refs[:,0], per_ref=False)

        losses = {}

        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                predict = getattr(self, f'predictor{i}')(out)
                predict_cross_video = getattr(self, f'predictor{i}')(out_cross_video)
                loss = self.ce_loss(predict, out_ind[idx])
                loss_cross_video = self.ce_loss(predict_cross_video, out_ind[idx])
                losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
                losses[f'ce{i}_cross_loss'] = (loss_cross_video * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]

        if self.l1_loss:
            num_feat = att_cross_video.shape[1]
            att_cross_video = att_cross_video.reshape(bsz, num_feat, -1, num_feat).permute(0, 2, 1, 3)
            label = torch.zeros(*att_cross_video.shape).cuda()
            m = torch.eye(bsz).cuda()
            m = (1 - m)[:,:,None,None].repeat(1,1,*att_cross_video.shape[-2:])
            losses['l1_loss'] = self.l1_loss(att_cross_video, label, weight=m)

        return losses
    
    
    

@MODELS.register_module()
class Vqvae_Tracker_V9(Vqvae_Tracker):
    """
    Args:
        Vqvae_Tracker ([type]): add a ce_loss on target frame
    """        
    
    def forward_train(self, imgs, mask_query_idx, jitter_imgs=None):
        
        bsz, num_clips, t, c, h, w = imgs.shape

        # vqvae tokenize for query frame
        with torch.no_grad():
            out_ind= []
            out_quant = []
            for i in range(self.num_head):
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc = getattr(self, f'vq_enc{i}')
                vqvae.eval()
                emb, quant, _, ind, _ = vq_enc(imgs[:, 0, -1])
                
                if self.per_ref:
                    ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                    quant = quant.unsqueeze(1).repeat(1, t-1, 1, 1, 1).permute(0,1,3,4,2).flatten(0,3).detach()
                    mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)
                else:
                    ind = ind.reshape(-1, 1).long().detach()
                    quant = quant.permute(0,2,3,1).flatten(0,2).detach()
                    mask_query_idx = mask_query_idx.bool()
                    
                out_ind.append(ind)
                out_quant.append(quant)

        if jitter_imgs is not None:
            imgs = jitter_imgs

        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            out, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            out, att = non_local_attention(tar, refs, per_ref=self.per_ref, temprature=self.temperature)

        losses = {}

        # if self.ce_loss:
        #     for idx, i in enumerate(range(self.num_head)):
        #         i = str(i).replace('0', '')
        #         if self.fc:
        #             predict = getattr(self, f'predictor{i}')(out)
        #         else:
        #             predict = self.embedding_layer(out)
        #             predict = nn.functional.normalize(predict, dim=-1)
        #             vq_emb = getattr(self, f'vq_emb{i}')
        #             predict = torch.mm(predict, nn.functional.normalize(vq_emb, dim=0))
        #             predict = torch.div(predict, 0.1) # temperature is set to 0.1
                    
        #         loss = self.ce_loss(predict, out_ind[idx])
        #         losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]

        predict_tar = self.predictor(tar.permute(0,2,3,1).flatten(0,2))
        loss_tar = self.ce_loss(predict_tar, out_ind[0])
        losses[f'target_ce_loss'] = (loss_tar * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()
        
        return losses
        

@MODELS.register_module()
class Vqvae_Tracker_V10(Vqvae_Tracker):
    """
    Args:
        Vqvae_Tracker ([type]): [induse long-term relationship]
    """        
    def __init__(self, soft_ce_loss, **kwargs):
        super().__init__(**kwargs)
        self.soft_ce_loss = build_loss(soft_ce_loss)
        self.fc = nn.Sequential(
            nn.Conv2d(512,2048,1),
            nn.ReLU(),
            nn.Conv2d(2048,512,1)
        )
    
    def forward_train(self, imgs, mask_query_idx, jitter_imgs=None):

        bsz, num_clips, t, c, h, w = imgs.shape

        # vqvae tokenize for query frame
        with torch.no_grad():
            out_ind= []
            for i in range(self.num_head):
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc = getattr(self, f'vq_enc{i}')
                vqvae.eval()
                emb, quant, _, ind, _ = vq_enc(imgs[:, 0, -1])
                
                if self.per_ref:
                    ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                    mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)
                else:
                    ind = ind.reshape(-1, 1).long().detach()
                    mask_query_idx = mask_query_idx.bool()
                
                out_ind.append(ind)

        if jitter_imgs is not None:
            imgs = jitter_imgs

        tar = self.backbone(imgs[:,0, 0])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(1,t)])
        
        # for short term
        out_s, att_s = non_local_attention(tar, [refs[0]], per_ref=self.per_ref)
        
        # for long term
        att_l = torch.eye(att_s.shape[-1]).repeat(bsz, 1, 1).cuda()
        refs.insert(0, tar)

        fs = torch.stack(refs, 1).flatten(0,1)
        fs = nn.functional.normalize(self.fc(fs), dim=1).permute(0, 2, 3, 1).reshape(bsz, t, att_s.shape[-1], -1)

        atts = torch.einsum('btic,btjc->btij',[fs[:,:-1], fs[:,1:]]) / 0.05
        atts_reverse = torch.flip(atts, [1]).permute(0,1,3,2)
        atts_cycle = torch.cat([atts, atts_reverse], 1)
        for i in range(atts_cycle.shape[1]):
            att = atts_cycle[:, i].softmax(dim=-1)
            att_l = torch.einsum('bij,bjk->bik', [att_l, att])
            
        del fs, refs, atts, atts_cycle, atts_reverse
        
        label = torch.arange(att_s.shape[-1]).repeat(bsz, 1).cuda().reshape(-1, 1)
            
        losses = {}
        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                predict_s = getattr(self, f'predictor{i}')(out_s)
                # loss_s = self.ce_loss(predict_s, out_ind[idx])
                # loss_l = self.soft_ce_loss(att_l.flatten(0,1), att_s.flatten(0,1).detach())
                loss_l = self.ce_loss(att_l.flatten(0,1), label)
                # losses[f'ce{i}_short_loss'] = (loss_s * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
                losses[f'ce{i}_long_loss'] = (loss_l * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()

        
        return losses
    



@MODELS.register_module()
class Vqvae_Tracker_V11(Vqvae_Tracker):
    """
    Args:
        Vqvae_Tracker ([type]): [induse long-term relationship  cycle-consistency]
    """        
    def __init__(self, soft_ce_loss, **kwargs):
        super().__init__(**kwargs)
        self.soft_ce_loss = build_loss(soft_ce_loss)
        self.fc = nn.Sequential(
            nn.Conv2d(512,2048,1),
            nn.ReLU(),
            nn.Conv2d(2048,512,1)
        )
        self._xent_targets = dict()
        self.edgedrop_rate = 0.1
        self.dropout = nn.Dropout(p=self.edgedrop_rate, inplace=False)
        
        # self.featdrop_rate = getattr(args, 'featdrop', 0)
        self.temperature = 0.07
        
    
    def infer_dims(self):
        in_sz = 256
        dummy = torch.zeros(1, 3, 1, in_sz, in_sz).to(next(self.encoder.parameters()).device)
        dummy_out = self.encoder(dummy)
        self.enc_hid_dim = dummy_out.shape[1]
        self.map_scale = in_sz // dummy_out.shape[-1]

    def make_head(self, depth=1):
        head = []
        if depth >= 0:
            dims = [self.enc_hid_dim] + [self.enc_hid_dim] * depth + [128]
            for d1, d2 in zip(dims, dims[1:]):
                h = nn.Linear(d1, d2)
                head += [h, nn.ReLU()]
            head = head[:-1]

        return nn.Sequential(*head)

    def zeroout_diag(self, A, zero=0):
        mask = (torch.eye(A.shape[-1]).unsqueeze(0).repeat(A.shape[0], 1, 1).bool() < 1).float().cuda()
        return A * mask

    def affinity(self, x1, x2):
        in_t_dim = x1.ndim
        if in_t_dim < 4:  # add in time dimension if not there
            x1, x2 = x1.unsqueeze(-2), x2.unsqueeze(-2)

        A = torch.einsum('bctn,bctm->btnm', x1, x2)
        # if self.restrict is not None:
        #     A = self.restrict(A)

        return A.squeeze(1) if in_t_dim < 4 else A
    
    def stoch_mat(self, A, zero_diagonal=False, do_dropout=True, do_sinkhorn=False):
        ''' Affinity -> Stochastic Matrix '''

        if zero_diagonal:
            A = self.zeroout_diag(A)

        if do_dropout and self.edgedrop_rate > 0:
            A[torch.rand_like(A) < self.edgedrop_rate] = -1e20

        return F.softmax(A/self.temperature, dim=-1)
    
    
    def xent_targets(self, A):
        B, N = A.shape[:2]
        key = '%s:%sx%s' % (str(A.device), B,N)

        if key not in self._xent_targets:
            I = torch.arange(A.shape[-1])[None].repeat(B, 1)
            self._xent_targets[key] = I.view(-1).to(A.device)

        return self._xent_targets[key]

    def forward_train_cycle(self, q, mask_query_idx=None):
        '''
        Input is B x T x N*C x H x W, where either
           N>1 -> list of patches of images
           N=1 -> list of images
        '''
        bsz, T, C, H, W = q.shape
        
        q = q.flatten(0,1)
        q = nn.functional.normalize(self.fc(q), dim=1).reshape(bsz, T, q.shape[1], -1).permute(0,2,1,3)
        # q = q.reshape(bsz, T, q.shape[1], -1).permute(0,2,1,3)
        
        #################################################################
        # Compute walks 
        #################################################################
        walks = dict()
        As = self.affinity(q[:, :, :-1], q[:, :, 1:])
        A12s = [self.stoch_mat(As[:, i], do_dropout=True) for i in range(T-1)]

        #################################################### Palindromes
        A21s = [self.stoch_mat(As[:, i].transpose(-1, -2), do_dropout=True) for i in range(T-1)]
        AAs = []
        for i in list(range(1, len(A12s))):
            g = A12s[:i+1] + A21s[:i+1][::-1]
            aar = aal = g[0]
            for _a in g[1:]:
                aar, aal = aar @ _a, _a @ aal

            AAs.append((f"r{i}", aar))

        for i, aa in AAs:
            walks[f"cyc {i}"] = [aa, self.xent_targets(aa)]


        #################################################################
        # Compute loss 
        #################################################################
        xents = [torch.tensor([0.]).cuda()]
        losses = dict()

        for name, (A, target) in walks.items():
            logits = torch.log(A+1e-20).flatten(0, -2)
            if mask_query_idx == None:
                loss = self.ce_loss(logits, target.unsqueeze(1)).mean()
            else:
                loss = self.ce_loss(logits, target.unsqueeze(1))
                loss = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()
            acc = (torch.argmax(logits, dim=-1) == target).float().mean()
            # diags.update({f"{H} xent {name}": loss.detach(),
            #               f"{H} acc {name}": acc})
            xents += [loss]


        loss = sum(xents)/max(1, len(xents)-1)
        
        return loss


    def forward_train(self, imgs, mask_query_idx, jitter_imgs=None):
    
        bsz, num_clips, t, c, h, w = imgs.shape

        # self.eval()
        
        # vqvae tokenize for query frame
        with torch.no_grad():
            out_ind= []
            for i in range(self.num_head):
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc = getattr(self, f'vq_enc{i}')
                vqvae.eval()
                emb, quant, _, ind, _ = vq_enc(imgs[:, 0, -1])
                
                if self.per_ref:
                    ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                    mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)
                else:
                    ind = ind.reshape(-1, 1).long().detach()
                    mask_query_idx = mask_query_idx.bool()
                
                out_ind.append(ind)

        if jitter_imgs is not None:
            imgs = jitter_imgs

        tar = self.backbone(imgs[:,0, 0])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(1,t)])

        
        # for short term
        out_s, att_s = non_local_attention(tar, [refs[0]], per_ref=self.per_ref)
     
        losses = {}
        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                predict_s = getattr(self, f'predictor{i}')(out_s)
                loss = self.ce_loss(predict_s, out_ind[idx])
                losses['ce_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()

        # for long term
        refs.insert(0, tar)
        all_feats = torch.stack(refs,1)
        losses['cycle_loss'] = self.forward_train_cycle(all_feats, mask_query_idx)
        
        # self.train()
        
        return losses
    
    

@MODELS.register_module()
class Vqvae_Tracker_V12(Vqvae_Tracker):
    
    def forward_train(self, imgs, mask_query_idx, frames_mask, jitter_imgs=None):
        
        bsz, num_clips, t, c, h, w = imgs.shape
    
        # vqvae tokenize for query frame
        with torch.no_grad():
            out_ind= []
            out_quant = []
            for i in range(self.num_head):
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc = getattr(self, f'vq_enc{i}')
                vqvae.eval()
                emb, quant, _, ind, _ = vq_enc(imgs[:, 0, -1])
                
                if self.per_ref:
                    ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                    quant = quant.unsqueeze(1).repeat(1, t-1, 1, 1, 1).permute(0,1,3,4,2).flatten(0,3).detach()
                    mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)
                else:
                    ind = ind.reshape(-1, 1).long().detach()
                    quant = quant.permute(0,2,3,1).flatten(0,2).detach()
                    mask_query_idx = mask_query_idx.bool()
                    
                out_ind.append(ind)
                out_quant.append(quant)

        if jitter_imgs is not None:
            imgs = jitter_imgs

        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            out, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            out, att = non_local_attention(tar, refs, per_ref=self.per_ref, temprature=self.temperature)

        losses = {}

        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                if self.fc:
                    predict = getattr(self, f'predictor{i}')(out)
                else:
                    predict = self.embedding_layer(out)
                    predict = nn.functional.normalize(predict, dim=-1)
                    vq_emb = getattr(self, f'vq_emb{i}')
                    predict = torch.mm(predict, nn.functional.normalize(vq_emb, dim=0))
                    predict = torch.div(predict, 0.1) # temperature is set to 0.1
                    
                loss = self.ce_loss(predict, out_ind[idx])
                losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
                
        if self.mse_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                predict = self.embedding_layer(out)
                loss = self.mse_loss(predict, out_quant[idx]).mean(-1)
                losses[f'mse{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
        
        mask = torch.ones(*att.shape).cuda() - frames_mask[:,:-1,None].flatten(3)
        target = torch.zeros(*att.shape).cuda()
        loss_reg = F.l1_loss(mask*att, target, reduction='none')
        losses['att_sparse_flow_loss'] = 10 * (loss_reg * mask_query_idx.unsqueeze(-1)).sum() / mask_query_idx.sum()
                    
        return losses
    
    
@MODELS.register_module()
class Vqvae_Tracker_V13(Vqvae_Tracker):
    
    def forward_train(self, imgs, mask_query_idx, frames_mask, jitter_imgs=None):
        
        bsz, num_clips, t, c, h, w = imgs.shape
    
        # vqvae tokenize for query frame
        with torch.no_grad():
            out_ind= []
            out_quant = []
            for i in range(self.num_head):
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc = getattr(self, f'vq_enc{i}')
                vqvae.eval()
                emb, quant, _, ind, _ = vq_enc(imgs[:, 0, -1])
                
                if self.per_ref:
                    ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                    quant = quant.unsqueeze(1).repeat(1, t-1, 1, 1, 1).permute(0,1,3,4,2).flatten(0,3).detach()
                    mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)
                else:
                    ind = ind.reshape(-1, 1).long().detach()
                    quant = quant.permute(0,2,3,1).flatten(0,2).detach()
                    mask_query_idx = mask_query_idx.bool()
                    
                out_ind.append(ind)
                out_quant.append(quant)

        if jitter_imgs is not None:
            imgs = jitter_imgs

        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])
        mask = frames_mask[:,:-1,None].flatten(3)

        if self.patch_size != -1:
            out, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            out, att = non_local_attention(tar, refs, per_ref=self.per_ref, temprature=self.temperature, mask=mask)
        
        losses = {}
        
        # a = att[0,0,0].detach().cpu().reshape(32,32).numpy()
        # b = mask[0,0,0].detach().cpu().reshape(32,32).numpy()

        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                if self.fc:
                    predict = getattr(self, f'predictor{i}')(out)
                else:
                    predict = self.embedding_layer(out)
                    predict = nn.functional.normalize(predict, dim=-1)
                    vq_emb = getattr(self, f'vq_emb{i}')
                    predict = torch.mm(predict, nn.functional.normalize(vq_emb, dim=0))
                    predict = torch.div(predict, 0.1) # temperature is set to 0.1
                    
                loss = self.ce_loss(predict, out_ind[idx])
                losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
                    
        return losses
    
    
@MODELS.register_module()
class Vqvae_Tracker_V15(Vqvae_Tracker):
    def __init__(self, vq_sample=True, *args, **kwargs):
        """ A video colorization-like vqvae_tracker. Refers to Video Colorization (ECCV 2017)

        Args:
            vq_sample (bool, optional): [description]. Defaults to True.
        """
        super().__init__(*args, **kwargs)
        
        self.vq_sample = vq_sample
        
        
    def forward_train(self, imgs, mask_query_idx, frames_mask=None, jitter_imgs=None):
        
        bsz, num_clips, t, c, h, w = imgs.shape
    
        # vqvae tokenize for query frame
        with torch.no_grad():
            out_ind= []
            out_quants = []
            vq_inds = []
            for i in range(self.num_head):
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc = getattr(self, f'vq_enc{i}')
                vqvae.eval()
                emb, quants, _, inds, _ = vq_enc(imgs.flatten(0,2))
                
                quants = quants.reshape(bsz, t, *quants.shape[-3:])
                inds = inds.reshape(bsz, t, *inds.shape[-2:])
                vq_inds.append([inds[:, -1], inds[:, -2]])
                
                if self.per_ref:
                    ind = inds[:, -1].unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                else:
                    ind = inds[:, -1].reshape(-1, 1).long().detach()
                    
                out_ind.append(ind)
                out_quant_refs = quants[:, :-1].flatten(3).permute(0, 1, 3, 2)
                out_quants.append(out_quant_refs)
            
        if jitter_imgs is not None:
            imgs = jitter_imgs

        # use pre-difined query
        mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1) if self.per_ref else mask_query_idx.bool()
        
        fs = self.backbone(imgs.flatten(0,2))
        if self.post_convolution is not None:
            fs = self.post_convolution(fs)
        if self.norm:
            fs = F.normalize(fs, dim=1)
        
        fs = fs.reshape(bsz, t, *fs.shape[-3:])
        tar, refs = fs[:, -1], fs[:, :-1]
        
        if frames_mask is not None:
            mask = frames_mask[:,:-1,None].flatten(3)
        else:
            mask = self.mask if self.temp_window else None

            
        if self.patch_size != -1:
            _, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            _, att = non_local_attention(tar, refs, per_ref=self.per_ref, temprature=self.temperature, mask=mask, scaling=self.scaling_att)
        
        losses = {}
                
        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):                
                i = str(i).replace('0', '')
                
                # change query if use vq sample
                if self.vq_sample:
                    mask_query_idx = self.query_vq_sample(vq_inds[idx][0], vq_inds[idx][1], t, self.mask, self.per_ref)
                
                out = frame_transform(att, out_quants[idx], per_ref=self.per_ref)
                
                if self.fc:
                    predict = getattr(self, f'predictor{i}')(out)
                else:
                    vq_emb = getattr(self, f'vq_emb{i}')
                    out = F.normalize(out, dim=-1)
                    vq_emb = F.normalize(vq_emb, dim=0)
                    predict = torch.mm(out, vq_emb)
                    # predict = torch.div(predict, 0.1) # temperature is set to 0.1
                    
                loss = self.ce_loss(predict, out_ind[idx])
                losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
                    
        return losses
    
    @staticmethod
    def query_vq_sample(ind_tar, ind_ref, t, mask, per_ref):
        # determined query
        ind_tar = ind_tar.flatten(1).unsqueeze(-1).repeat(1,1,ind_tar.shape[-1] * ind_tar.shape[-2])
        ind_ref = ind_ref.flatten(1).unsqueeze(1).repeat(1,ind_ref.shape[-1] * ind_ref.shape[-2],1)
        mask_query_idx = ((ind_tar == ind_ref) * mask.unsqueeze(0)).sum(-1)
        mask_query_idx = (mask_query_idx > 0)
        if per_ref:
            mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)

        return mask_query_idx
    
    
    

@MODELS.register_module()
class Vqvae_Tracker_V16(Vqvae_Tracker_V15):
    '''  Combine with MAST CVPR2020 '''
    
    def __init__(self, l1_loss=False, downsample_rate=8, downsample_mode='default', use_quant=True, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.downsample_rate = downsample_rate
        self.l1_loss = l1_loss
        self.downsample_mode = downsample_mode
        self.use_quant = use_quant
    
    def prep(self, image, mode='default'):
        bsz,c,_,_ = image.size()
        if mode == 'default':
            x = image.float()[:,:,::self.downsample_rate,::self.downsample_rate]
        elif mode == 'unfold':
            x = F.unfold(image, self.downsample_rate+1, padding=self.downsample_rate//2)
            x = x.reshape(bsz, -1, *image.shape[-2:]).mean(1, keepdim=True)
            x = x.float()[:,:,::self.downsample_rate,::self.downsample_rate]
        return x
        
    def forward_train(self, imgs, mask_query_idx, frames_mask=None, images_lab=None):
        
        bsz, num_clips, t, c, h, w = imgs.shape
        
        # use pre-difined query
        mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1) if self.per_ref else mask_query_idx.bool()
        
        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        if self.post_convolution is not None:
            fs = self.post_convolution(fs)
        if self.norm:
            fs = F.normalize(fs, dim=1)
        fs = fs.reshape(bsz, t, *fs.shape[-3:])
        tar, refs = fs[:, -1], fs[:, :-1]
        
        # get correlation attention map
        if frames_mask is not None:
            mask = frames_mask[:,:-1,None].flatten(3)
        else:
            mask = self.mask if self.temp_window else None
                        
        if self.patch_size != -1:
            _, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            _, att = non_local_attention(tar, refs, per_ref=self.per_ref, temprature=self.temperature, mask=mask, scaling=self.scaling_att)
    
        losses = {}
        
        # for mast l1_loss
        if self.l1_loss:
            ref_gt = [self.prep(gt[:,ch], mode=self.downsample_mode) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs.shape[-2:])     
            losses['l1_loss'], _ = self.compute_lphoto(images_lab_gt, ch, outputs)
        
        # for ce_loss
        if self.ce_loss:
            # vqvae tokenize for query frame
            with torch.no_grad():
                out_ind= []
                outs = []
                vq_inds = []
                for i in range(self.num_head):
                    i = str(i).replace('0', '')
                    vqvae = getattr(self, f'vqvae{i}')
                    vq_enc = getattr(self, f'vq_enc{i}')
                    vqvae.eval()
                    encs, quants, _, inds, _ = vq_enc(imgs.flatten(0,2))
                    
                    if self.use_quant:
                        quants = quants.reshape(bsz, t, *quants.shape[-3:])
                        out_quant_refs = quants[:, :-1].flatten(3).permute(0, 1, 3, 2)
                        outs.append(out_quant_refs)
                    else:
                        encs = encs.reshape(bsz, t, *encs.shape[-3:])
                        out_enc_refs = encs[:, :-1].flatten(3).permute(0, 1, 3, 2)
                        outs.append(out_enc_refs)
                    
                    inds = inds.reshape(bsz, t, *inds.shape[-2:])
                    vq_inds.append([inds[:, -1], inds[:, -2]])
                    
                    if self.per_ref:
                        ind = inds[:, -1].unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                    else:
                        ind = inds[:, -1].reshape(-1, 1).long().detach()
                        
                    out_ind.append(ind)

            
            for idx, i in enumerate(range(self.num_head)):                
                i = str(i).replace('0', '')
                
                # change query if use vq sample
                if self.vq_sample:
                    mask_query_idx = self.query_vq_sample(vq_inds[idx][0], vq_inds[idx][1], t, self.mask, self.per_ref)
                
                out = frame_transform(att, outs[idx], per_ref=self.per_ref)
                
                if self.fc:
                    predict = getattr(self, f'predictor{i}')(out)
                else:
                    vq_emb = getattr(self, f'vq_emb{i}')
                    out = F.normalize(out, dim=-1)
                    vq_emb = F.normalize(vq_emb, dim=0)
                    predict = torch.mm(out, vq_emb)
                    # predict = torch.div(predict, 0.1) # temperature is set to 0.1
                    
                loss = self.ce_loss(predict, out_ind[idx])
                losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
                    
        return losses
    
    @staticmethod
    def query_vq_sample(ind_tar, ind_ref, t, mask, per_ref):
        # determined query
        ind_tar = ind_tar.flatten(1).unsqueeze(-1).repeat(1,1,ind_tar.shape[-1] * ind_tar.shape[-2])
        ind_ref = ind_ref.flatten(1).unsqueeze(1).repeat(1,ind_ref.shape[-1] * ind_ref.shape[-2],1)
        mask_query_idx = ((ind_tar == ind_ref) * mask.unsqueeze(0)).sum(-1)
        mask_query_idx = (mask_query_idx > 0)
        if per_ref:
            mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)

        return mask_query_idx
    
    def dropout2d_lab(self, arr): # drop same layers for all images
        if not self.training:
            return arr

        drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
        drop_ch_ind = np.random.choice(np.arange(1,3), drop_ch_num, replace=False)

        for a in arr:
            for dropout_ch in drop_ch_ind:
                a[:, dropout_ch] = 0
            a *= (3 / (3 - drop_ch_num))

        return arr, drop_ch_ind # return channels not masked
    
    def compute_lphoto(self, images_lab_gt, ch, outputs):
        b, c, h, w = images_lab_gt[0].size()

        if self.downsample_mode == 'default':
            tar_y = images_lab_gt[-1][:,ch]  # y4
            outputs = F.interpolate(outputs, (h, w), mode='bilinear')
        else:
            tar_y = self.prep(images_lab_gt[-1][:,ch], mode=self.downsample_mode)
        
        loss = F.smooth_l1_loss(outputs*20, tar_y*20, reduction='mean')

        err_maps = torch.abs(outputs - tar_y).sum(1).detach()

        return loss, err_maps