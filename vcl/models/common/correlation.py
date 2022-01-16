import torch.nn.functional as F
import torch


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


def non_local_attention(tar, refs, per_ref=True, flatten=True, temprature=1.0):
    
    """ Given refs and tar, return transform tar non-local.

    Returns:
        att: attention for tar wrt each ref (concat) 
        out: transform tar for each ref if per_ref else for all refs
    """
    
    if isinstance(refs, list):
        refs = torch.stack(refs, 1)

    bsz, t, feat_dim, w_, h_ = refs.shape

    tar = tar.flatten(2).permute(0, 2, 1)
    refs = refs.flatten(3).permute(0, 1, 3, 2)
    
    att = torch.einsum("bic,btjc -> btij", (tar, refs)) / temprature
    
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
    
    if local:
        assert patch_size != -1
        bsz, t, feat_dim, w_, h_ = refs.shape
        unfold_fs = list([ F.unfold(ref, kernel_size=patch_size, \
            padding=int((patch_size-1)/2)).reshape(bsz, feat_dim, -1, w_, h_) for ref in refs])
        unfold_fs = torch.cat(unfold_fs, 2)

        out = (unfold_fs * att).sum(2).reshape(bsz, feat_dim, -1).permute(0,2,1).reshape(-1, feat_dim)
                                                                                         
    else:
        
        if not per_ref:
            out =  torch.einsum('bij,bjc -> bic', [att, refs.flatten(1,2)])
        else:
            # "btij,btjc -> btic“
            out = torch.matmul(att, refs)
            
    if flatten:
        out = out.reshape(-1, refs.shape[-1])
        
    return out
        