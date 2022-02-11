import torch.nn.functional as F
import torch
import torch.nn as nn
from spatial_correlation_sampler import SpatialCorrelationSampler
from .utils import *

def local_attention(correlation_sampler, tar, refs, patch_size):
    
    """ Given refs and tar, return transform tar local.

    Returns:
        att: attention for tar wrt each ref (concat) 
        out: transform tar for each ref
    """
    bsz, feat_dim, w_, h_ = tar.shape
    t = len(refs)
    
    corrs = []
    for i in range(t-1):
        corr = correlation_sampler(tar.contiguous(), refs[i].contiguous()).reshape(bsz, -1, w_, h_)
        corrs.append(corr)

    corrs = torch.cat(corrs, 1)
    att = F.softmax(corrs, 1).unsqueeze(1)
    
    out = frame_transform(att, refs, local=True, patch_size=patch_size)

    return out, att


def non_local_attention(tar, refs, per_ref=True, flatten=True, temprature=1.0, mask=None):
    
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

    if mask is not None:
        att *= mask
    
    if per_ref:
        # return att for each ref
        att = F.softmax(att, dim=-1)
        out = frame_transform(att, refs, per_ref=per_ref, flatten=flatten)
        return out, att  
    else:
        att_ = att.permute(0, 2, 1, 3).flatten(2)
        att_ = F.softmax(att_, -1)
        out = frame_transform(att_, refs, per_ref=per_ref, flatten=flatten)
        return out, att_


def inter_intra_attention(tar, refs, per_ref=True, flatten=True, temprature=1.0, mask=None):
    
    if isinstance(refs, list):
        refs = torch.stack(refs, 1)

    tar = tar.flatten(2).permute(0, 2, 1)
    
    _, feat_dim, w_, h_ = refs.shape
    refs = refs.flatten(2).permute(0, 2, 1)
    att = torch.einsum("bic,djc -> bdij", (tar, refs)) / temprature
    
    if mask is not None:
        att *= mask
    
    if per_ref:
        # return att for each ref
        att = F.softmax(att, dim=-1)
        out = frame_transform(att, refs, per_ref=per_ref, flatten=flatten)
        return out, att  
    else:
        att_ = att.permute(0, 2, 1, 3).flatten(2)
        att_ = F.softmax(att_, -1)
        out = frame_transform(att_, refs, per_ref=per_ref, flatten=flatten)
        return out, att_

def frame_transform(att, refs, per_ref=True, local=False, patch_size=-1, flatten=True):
    
    """transform a target frame given refs and att

    Returns:
        out: transformed feature map (B*T*H*W) x C  if per_ref else (B*H*W) x C
        
    """
    if isinstance(refs, list):
        refs = torch.stack(refs, 1)
        refs = refs.flatten(3).permute(0, 1, 3, 2)
        
    if local:
        assert patch_size != -1
        bsz, t, feat_dim, w_, h_ = refs.shape
        unfold_fs = list([ F.unfold(ref, kernel_size=patch_size, \
            padding=int((patch_size-1)/2)).reshape(bsz, feat_dim, -1, w_, h_) for ref in refs])
        unfold_fs = torch.cat(unfold_fs, 2)
        out = (unfold_fs * att).sum(2).reshape(bsz, feat_dim, -1).permute(0,2,1).reshape(-1, feat_dim)                                                                          
    else:
        if not per_ref:
            if refs.dim() == 4: 
                out =  torch.einsum('bij,bjc -> bic', [att, refs.flatten(1,2)])
            else:
                out =  torch.einsum('bij,jc -> bic', [att, refs.flatten(0,1)])
        else:
            # "btij,btjc -> btic“
            out = torch.matmul(att, refs)
            
    if flatten:
        out = out.reshape(-1, refs.shape[-1])
        
    return out

class Colorizer(nn.Module):
    def __init__(self, D=4, R=6, C=32):
        super(Colorizer, self).__init__()
        self.D = D
        self.R = R  # window size
        self.C = C

        self.P = self.R * 2 + 1
        self.N = self.P * self.P
        self.count = 0

        self.memory_patch_R = 12
        self.memory_patch_P = self.memory_patch_R * 2 + 1
        self.memory_patch_N = self.memory_patch_P * self.memory_patch_P

        self.correlation_sampler_dilated = [
            SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.memory_patch_P,
            stride=1,
            padding=0,
            dilation=1,
            dilation_patch=dirate) for dirate in range(2,6)]

        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.P,
            stride=1,
            padding=0,
            dilation=1)

    def prep(self, image, HW):
        _,c,_,_ = image.size()

        x = image.float()[:,:,::self.D,::self.D]

        if c == 1 and not self.training:
            x = one_hot(x.long(), self.C)

        return x

    def forward(self, feats_r, feats_t, quantized_r, ref_index, current_ind, dil_int = 15):
        """
        Warp y_t to y_(t+n). Using similarity computed with im (t..t+n)
        :param feats_r: f([im1, im2, im3])
        :param quantized_r: [y1, y2, y3]
        :param feats_t: f(im4)
        :param mode:
        :return:
        """
        # For frame interval < dil_int, no need for deformable resampling
        nref = len(feats_r)
        nsearch = len([x for x in ref_index if current_ind - x > dil_int])

        # The maximum dilation rate is 4
        dirates = [ min(4, (current_ind - x) // dil_int +1) for x in ref_index if current_ind - x > dil_int]
        b,c,h,w = feats_t.size()
        N = self.P * self.P
        corrs = []

        # offset0 = []
        for searching_index in range(nsearch):
            ##### GET OFFSET HERE.  (b,h,w,2)
            samplerindex = dirates[searching_index]-2
            coarse_search_correlation = self.correlation_sampler_dilated[samplerindex](feats_t, feats_r[searching_index])  # b, p, p, h, w
            coarse_search_correlation = coarse_search_correlation.reshape(b, self.memory_patch_N, h*w)
            coarse_search_correlation = F.softmax(coarse_search_correlation, dim=1)
            coarse_search_correlation = coarse_search_correlation.reshape(b,self.memory_patch_P,self.memory_patch_P,h,w,1)
            _y, _x = torch.meshgrid(torch.arange(-self.memory_patch_R,self.memory_patch_R+1),torch.arange(-self.memory_patch_R,self.memory_patch_R+1))
            grid = torch.stack([_x, _y], dim=-1).unsqueeze(-2).unsqueeze(-2)\
                .reshape(1,self.memory_patch_P,self.memory_patch_P,1,1,2).contiguous().float().to(coarse_search_correlation.device)
            offset0 = (coarse_search_correlation * grid ).sum(1).sum(1) * dirates[searching_index]  # 1,h,w,2

            col_0 = deform_im2col(feats_r[searching_index], offset0, kernel_size=self.P)  # b,c*N,h*w
            col_0 = col_0.reshape(b,c,N,h,w)
            ##
            corr = (feats_t.unsqueeze(2) * col_0).sum(1)   # (b, N, h, w)

            corr = corr.reshape([b, self.P * self.P, h * w])
            corrs.append(corr)

        for ind in range(nsearch, nref):
            corrs.append(self.correlation_sampler(feats_t, feats_r[ind]))
            _, _, _, h1, w1 = corrs[-1].size()
            corrs[ind] = corrs[ind].reshape([b, self.P*self.P, h1*w1])

        corr = torch.cat(corrs, 1)  # b,nref*N,HW
        corr = F.softmax(corr, dim=1)
        corr = corr.unsqueeze(1)

        qr = [self.prep(qr, (h,w)) for qr in quantized_r]

        im_col0 = [deform_im2col(qr[i], offset0, kernel_size=self.P)  for i in range(nsearch)]# b,3*N,h*w
        im_col1 = [F.unfold(r, kernel_size=self.P, padding =self.R) for r in qr[nsearch:]]
        image_uf = im_col0 + im_col1

        image_uf = [uf.reshape([b,qr[0].size(1),self.P*self.P,h*w]) for uf in image_uf]
        image_uf = torch.cat(image_uf, 2)
        out = (corr * image_uf).sum(2).reshape([b,qr[0].size(1),h,w])

        return out


def torch_unravel_index(indices, shape):
    rows = indices / shape[0]
    cols = indices % shape[1]

    return (rows, cols)