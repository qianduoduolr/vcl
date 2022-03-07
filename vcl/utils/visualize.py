import os
import numpy as np
import cv2
import torch

import mmcv
from PIL import Image
import matplotlib.pyplot as plt
from .util import *
from .dim_reduction import *
from matplotlib import cm


class Correspondence_Visualizer(object):
    def __init__(self, mode, show_mode='plt', flow_show_mode='rgb', nembed=2048, scale=32):
        
        self.mode = mode
        self.show_mode = show_mode
        self.flow_show_mode = flow_show_mode
        self.scale = scale
        
        if self.show_mode == 'plt':
            plt.rcParams['figure.dpi'] = 200
        
        if self.mode == 'vq':
            if nembed >= 256:
                self.rescale = True
            self.nembed = nembed


    def vis_pairwise_attention(self, frames, fs, sample_idx):
        
        att = affanity(fs[0], [fs[1]], flatten=False)
        scale = int(att.shape[-1] ** 0.5)
        att = att[0, 0, sample_idx].reshape(scale,scale).detach().cpu().numpy() * 255
        att = self.att_norm(att)
        
        x, y = sample_idx % scale, sample_idx // scale
        querys_map = np.zeros((scale,scale))
        querys_map[y,x] = 255
        querys_map = querys_map.astype(np.uint8)
        
        # return blend result
        blend_out_result, blend_out_query = self.att_blend(att, querys_map, frames)
        
        # show
        if self.show_mode == 'plt':
            plt.figure()
            plt.subplot(1,4,1), plt.imshow(blend_out_query)
            plt.subplot(1,4,2), plt.imshow(blend_out_result)
            plt.subplot(1,4,3), plt.imshow(querys_map)
            plt.subplot(1,4,4), plt.imshow(att)
        else:
            pass
        
    def vis_vq(self, z1_q, z2_q, frame1, frame2, x_rec1=None, x_rec2=None):
            
        frame_vq_1 = z1_q.permute(1,2,0).numpy()
        frame_vq_2 = z2_q.permute(1,2,0).numpy()

        print(len(np.unique(frame_vq_1)))

        if self.rescale:
            frame_vq_1 = (frame_vq_1 * 255 / self.nembed).astype(np.uint8)
            frame_vq_2 = (frame_vq_2 * 255 / self.nembed).astype(np.uint8)
        else:
            frame_vq_1 = (frame_vq_1).astype(np.uint8)
            frame_vq_2 = (frame_vq_2).astype(np.uint8)

        if self.show_mode == 'plt':
            plt.figure()

            if x_rec1 is not None:
                plt.subplot(3,2,1), plt.imshow(frame_vq_1, cmap=plt.get_cmap('jet'))
                plt.subplot(3,2,2), plt.imshow(frame_vq_2, cmap=plt.get_cmap('jet'))
                plt.subplot(3,2,3), plt.imshow(np.array(frame1))
                plt.subplot(3,2,4), plt.imshow(np.array(frame2))
                plt.subplot(3,2,5), plt.imshow(np.array(x_rec1))
                plt.subplot(3,2,6), plt.imshow(np.array(x_rec2))
                plt.show()
            else:
                plt.subplot(2,2,1), plt.imshow(frame_vq_1, cmap=plt.get_cmap('jet'))
                plt.subplot(2,2,2), plt.imshow(frame_vq_2, cmap=plt.get_cmap('jet'))
                plt.subplot(2,2,3), plt.imshow(np.array(frame1))
                plt.subplot(2,2,4), plt.imshow(np.array(frame2))
        else:
            pass

    def vis_pca(self, fs, frames):
        
        x = torch.cat(fs, 0)
        
        pca_ff = pca_feats(x)
        pca_ff = pca_ff.permute(0, 2, 3, 1)
                
        pca_ff1 = cv2.resize(pca_ff[0].numpy(), (frames[0].shape[0], frames[0].shape[1]))
        pca_ff2 = cv2.resize(pca_ff[1].numpy(), (frames[1].shape[0], frames[1].shape[1]))
        
        if self.show_mode == 'plt':
            plt.figure()
            plt.subplot(1,4,1), plt.imshow(frames[0])
            plt.subplot(1,4,2), plt.imshow(frames[1])
            plt.subplot(1,4,3), plt.imshow(pca_ff1)
            plt.subplot(1,4,4), plt.imshow(pca_ff2)
        else:
            pass
    
    def vis_flow(self, fs, xs, frames, gt=None):
        
        att = affanity(fs[0], [fs[1]], flatten=False)
        att = att.detach()
        
        u, v = compute_flow(att[:,0])
        flow_visualize(u, v, xs[0][0], xs[1][0], att[0,0], frames, gt=gt, mode=self.flow_show_mode)

    def vis_multiple_frrames_attention(self):
        pass
    
    
    def visualize(self, *args, **kwargs):
        if self.mode == 'pair':
            self.vis_pairwise_attention(*args, **kwargs)
        elif self.mode == 'vq':
            self.vis_vq(*args, **kwargs)
        elif self.mode == 'pca':
            self.vis_pca(*args, **kwargs)
        elif self.mode == 'flow':
            self.vis_flow(*args, **kwargs)
            
            
    @staticmethod
    def att_norm(att, dim=None):
        att = ((att - att.min()) * 255 / (att.max() - att.min())).astype(np.uint8)
        return att

    @staticmethod
    def att_blend(att, querys_map, frames):
        h, w = frames[-1].shape[0:2]
        
        att = cv2.resize(att, (h, w))
        querys_map = cv2.resize(querys_map, (h, w))

        att_map = cv2.applyColorMap(att.astype(np.uint8), cv2.COLORMAP_BONE)
        att_query_map = cv2.applyColorMap(querys_map, cv2.COLORMAP_BONE)


        tar = Image.fromarray(frames[0]).convert('RGBA')
        query = Image.fromarray(att_query_map).convert('RGBA')
        blend_out_query = Image.blend(tar, query, 0.7)
        blend_out_query = np.array(blend_out_query)

        ref = Image.fromarray(frames[1]).convert('RGBA')
        result = Image.fromarray(att_map).convert('RGBA')
        blend_out_result = Image.blend(ref, result, 0.7)
        blend_out_result = np.array(blend_out_result)
        
        return blend_out_result, blend_out_query

     
    def save(self, frames, att):
        pass
    
    
    
def preprocess_(img, mode='rgb'):
    
    if mode == 'rgb':
        mean=np.array([123.675, 116.28, 103.53])
        std=np.array([58.395, 57.12, 57.375])
    else:
        mean=np.array([50, 0, 0])
        std=np.array([50, 127, 127])
        img = (img / 255.0).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    
    # resize
    img = img.astype(np.float32)
    mmcv.imnormalize_(img, mean, std, False)
    out = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
    
    return out
    
    
def compute_flow(corr):
    # assume batched affinity, shape N x H * W x W x H
    h = w = int(corr.shape[-1] ** 0.5)

    # x1 -> x2
    corr = corr.transpose(-1, -2).view(*corr.shape[:-1], h, w)
    nnf = corr.argmax(dim=1)

    u = nnf % w # nnf.shape[-1]
    v = nnf / h # nnf.shape[-2] # nnf is an IntTensor so rounds automatically

    rr = torch.arange(u.shape[-1])[None].long().cuda()

    for i in range(u.shape[-1]):
        u[:, i] -= rr

    for i in range(v.shape[-1]):
        v[:, :, i] -= rr

    return u, v


def flow_visualize(u, v, x1, x2, A, frames=None, gt=None, mode='rgb'):
    flows = torch.stack([u, v], dim=-1).cpu().numpy()
        
    if mode == 'arrow':   
        plt.figure()
        plt.subplot(1,3,1), plt.imshow(frames[0])
        plt.subplot(1,3,2), plt.imshow(frames[1])
        
        I, flows = x1.cpu().numpy(), flows[0]
        H, W = flows.shape[:2]
        Ih, Iw, = I.shape[-2:]
        mx, my = np.mgrid[0:Ih:Ih/(H+1), 0:Iw:Iw/(W+1)][:, 1:, 1:]
        skip = (slice(None, None, 1), slice(None, None, 1))

        ii = 0
        fig, ax = plt.subplots()
        # im = ax.imshow((I.transpose(1,2,0)),)
        C = cm.jet(torch.nn.functional.softmax((A * A.log()).sum(-1).cpu(), dim=-1))
        # ax.quiver(my[skip], mx[skip], flows[...,0][skip], flows[...,1][skip]*-1, C)#, scale=1, scale_units='dots')
        ax.quiver(mx[skip], my[skip], flows[...,0][skip], flows[...,1][skip])
        plt.subplot(1,3,3), plt.imshow()
        
    else:
        flows = flows[0].astype(np.float32)
        H, W = flows.shape[:2]

        result = mmcv.visualization.flow2rgb(flows)
        
        plt.figure()
        plt.subplot(1,4,1), plt.imshow(frames[0])
        plt.subplot(1,4,2), plt.imshow(frames[1])
        plt.subplot(1,4,3), plt.imshow(result)
        
        if gt.any() == None:
            plt.subplot(1,4,4), plt.imshow(result)
        else:
            gt = mmcv.visualization.flow2rgb(gt)
            plt.subplot(1,4,4), plt.imshow(gt)


def affanity(tar, refs, per_ref=True, flatten=True, temprature=1.0, mask=None, scaling=False):
    
    """ Given refs and tar, return transform tar non-local.

    Returns:
        att: attention for tar wrt each ref (concat) 
        out: transform tar for each ref if per_ref else for all refs
    """
    
    if isinstance(refs, list):
        refs = torch.stack(refs, 1)

    tar = tar.flatten(2).permute(0, 2, 1)
    
    _, t, feat_dim, w_, h_ = refs.shape
    refs = refs.flatten(3).permute(0, 1, 3, 2)
    att = torch.einsum("bic,btjc -> btij", (tar, refs)) / temprature 
    if scaling:
        # scaling
        att = att / torch.sqrt(torch.tensor(feat_dim).float()) 

    if mask is not None:
        # att *= mask
        att.masked_fill_(~mask.bool(), float('-inf'))
    
    return att