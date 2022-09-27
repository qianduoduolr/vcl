import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from typing import List
import math, time

from vcl.utils.visualize import affanity

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """Efficient version of torch.cat that avoids a copy if there is only a
    single element in a list."""
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def coords_grid(batch: int, xx, yy):
    """Coordinate grid.
    Args:
        batch (int): The batch size of feature.
        xx (Tensor): 1-D tensor of size W with values from the interval
            [0, W-1].
        yy (Tensor): 1-D tensor of size H with values from the interval
            [0, H-1].
    Returns:
        Tensor: Tensor of shape (batch, 2, H, W) with values of items'
            coordinate.
    """
    coords = torch.meshgrid(yy, xx)
    coords = torch.stack(coords[::-1], dim=0).float()

    return coords[None].repeat(batch, 1, 1, 1)  # shape(batch, 2, H, W)


def local_square_attention(query,
                           key,
                           value,
                           kernel_size,
                           temperature=1,
                           topk=None,
                           batch_as_context=False):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, H, W)
        kernel_size (int | tuple[int]):
        temperature (float)
        topk (int)
        batch_as_context (bool): Take batches as context for key

    Returns:

    """
    assert query.ndim == key.ndim == 4
    assert query.shape[1:] == key.shape[1:]
    assert value.shape[2:] == key.shape[2:], f'{value.shape} {key.shape}'
    assert value.shape[0] == key.shape[0]
    channels, height, width = query.shape[1:]
    kernel_size = _pair(kernel_size)
    padding = tuple(k // 2 for k in kernel_size)
    # [N, Cxhxw, HxW]
    unfolded_key = F.unfold(key, kernel_size=kernel_size, padding=padding)
    unfolded_value = F.unfold(value, kernel_size=kernel_size, padding=padding)
    # [N, C, hxw, HxW]
    unfolded_key = unfolded_key.view(unfolded_key.shape[0], channels,
                                     kernel_size[0] * kernel_size[1],
                                     height * width)
    unfolded_value = unfolded_value.view(unfolded_value.shape[0],
                                         value.shape[1],
                                         kernel_size[0] * kernel_size[1],
                                         height * width)
    # [N, C, 1, HxW]
    unfolded_query = query.reshape(query.shape[0], channels,
                                   height * width).unsqueeze(2)
    if batch_as_context:
        # [1, C, Nxhxw, HxW]
        unfolded_key.transpose_(0, 1)
        unfolded_key = unfolded_key.reshape(
            1, channels, key.shape[0] * kernel_size[0] * kernel_size[1],
            height * width)
        unfolded_value.transpose_(0, 1)
        unfolded_value = unfolded_value.reshape(
            1, value.shape[1],
            value.shape[0] * kernel_size[0] * kernel_size[1], height * width)
    # [N, 1, hxw, HxW] or [N, 1, Nxhxw, HxW]
    attention = torch.zeros(query.shape[0], 1,
                            *unfolded_key.shape[2:]).to(unfolded_query)
    spatial_step = 512
    for ptr in range(0, height * width, spatial_step):
        attention[..., ptr:ptr + spatial_step] = torch.sum(
            unfolded_query[..., ptr:ptr + spatial_step] *
            unfolded_key[..., ptr:ptr + spatial_step],
            dim=1,
            keepdim=True)
    attention /= temperature
    # attention = torch.sum(unfolded_query * unfolded_key, dim=1,
    #                       keepdim=True) / temperature
    if topk is not None:
        topk_attention, topk_indices = attention.topk(k=topk, dim=2)
        # [N, 1, topk, HxW]
        attention = topk_attention
        # [N, C, topk, HxW]
        unfolded_value = unfolded_value.gather(
            dim=2, index=topk_indices.expand(-1, value.shape[1], -1, -1))
    # [N, C, HxW]
    output = torch.sum(attention * unfolded_value, dim=2)
    output = output.reshape(output.shape[0], output.shape[1], height, width)

    return output


def local_corr_attention(query,
                         key,
                         value,
                         kernel_size,
                         temperature=1,
                         topk=None,
                         batch_as_context=False):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, H, W)
        kernel_size (int | tuple[int]):
        temperature (float)
        topk (int)
        batch_as_context (bool): Take batches as context for key

    Returns:

    """
    # not tested
    from spatial_correlation_sampler import spatial_correlation_sample
    assert query.ndim == key.ndim == 4
    assert query.shape[1:] == key.shape[1:]
    assert value.shape[2:] == key.shape[2:], f'{value.shape} {key.shape}'
    assert value.shape[0] == key.shape[0]
    channels, height, width = query.shape[1:]
    kernel_size = _pair(kernel_size)
    padding = tuple(k // 2 for k in kernel_size)
    assert batch_as_context
    assert query.shape[0] == 1
    # [N, Cxhxw, HxW]
    unfolded_value = F.unfold(value, kernel_size=kernel_size, padding=padding)
    # [N, C, hxw, HxW]
    unfolded_value = unfolded_value.view(unfolded_value.shape[0],
                                         value.shape[1],
                                         kernel_size[0] * kernel_size[1],
                                         height * width)
    key_batch_size = key.shape[0]
    attentions = []
    for i in range(key_batch_size):
        # [1, h, w, H, W]
        attention = spatial_correlation_sample(
            query, key[i:i + 1], kernel_size=1, patch_size=kernel_size)
        attentions.append(attention)
    # [N, h, w, H, W]
    attentions = cat(attentions, dim=0)
    attentions /= temperature
    # [C, Nxhxw, HxW]
    unfolded_value.transpose_(0, 1)
    unfolded_value = unfolded_value.reshape(
        value.shape[1], value.shape[0] * kernel_size[0] * kernel_size[1],
        height * width)
    # [1, Nxhxw, HxW]
    attentions = attentions.view(
        1, key.shape[0] * kernel_size[0] * kernel_size[1], height * width)

    if topk is not None:
        topk_attentions, topk_indices = attentions.topk(k=topk, dim=1)
        # [1, topk, HxW]
        attentions = topk_attentions
        # [C, topk, HxW]
        unfolded_value = unfolded_value.gather(
            dim=1, index=topk_indices.expand(value.shape[1], -1, -1))
    # [C, 1, HxW]
    output = torch.einsum('cij,bij->cbj', attentions.softmax(dim=1),
                          unfolded_value)
    output.transpose_(0, 1)
    output = output.reshape(1, value.shape[1], height, width)

    return output


def masked_attention(query,
                     key,
                     value,
                     mask,
                     temperature=1,
                     topk=None,
                     normalize=True,
                     step=100):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, T, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, T, H, W)
        temperature (float)
        topk (int)
        normalize (bool)
        step (int)

    Returns:

    """
    batches = query.size(0)
    assert query.size(0) == key.size(0) == value.size(0)
    assert value.shape[2:] == key.shape[2:], f'{value.shape} {key.shape}'
    if key.ndim == 4:
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
    assert value.ndim == key.ndim == 5
    clip_len = key.size(2)
    assert query.shape[2:] == key.shape[3:]
    att_channels, height, width = query.shape[1:]
    C = value.size(1)
    if normalize:
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)
    query_vec = query.view(batches, att_channels, query.shape[2:].numel())
    key_vec = key.view(batches, att_channels, key.shape[2:].numel())
    value_vec = value.view(batches, C, value.shape[2:].numel())
    # [N, TxHxW, HxW]
    affinity = torch.einsum('bci,bcj->bij', key_vec, query_vec) / temperature
    mask = mask.view(1, height * width,
                     height * width).expand(clip_len, -1,
                                            -1).reshape_as(affinity)
    affinity.masked_fill_(~mask.bool(), float('-inf'))
    output = torch.zeros(batches, C, height * width).to(query)
    for ptr in range(0, height * width, step):
        # [N, TxHxW, step]
        cur_affinity = affinity[:, :, ptr:ptr + step]
        if topk is not None:
            # [N, topk, step]
            topk_affinity, topk_indices = cur_affinity.topk(k=topk, dim=1)
            # cur_affinity, idx = cur_affinity.sort(descending=True, dim=1)
            # topk_affinity, topk_indices = cur_affinity[:, :topk], idx[:,
            # :topk]
            # assert torch.allclose(topk_affinity, topk_affinity_)
            # assert torch.allclose(topk_indices, topk_indices_)
            topk_value = value_vec.transpose(0, 1).reshape(
                C, -1).index_select(
                    dim=1, index=topk_indices.reshape(-1))
            # [N, C, topk, step]
            topk_value = topk_value.reshape(C,
                                            *topk_indices.shape).transpose(
                                                0, 1)
            cur_output = torch.einsum('bcks,bks->bcs', topk_value,
                                      topk_affinity.softmax(dim=1))
        else:
            cur_output = torch.einsum('bck,bks->bcs', value_vec,
                                      cur_affinity.softmax(dim=1))
        output[..., ptr:ptr + step] = cur_output

    output = output.reshape(batches, C, height, width)

    return output


def masked_attention_efficient(query,
                               key,
                               value,
                               mask,
                               temperature=1,
                               topk=None,
                               normalize=True,
                               step=32,
                               non_mask_len=0,
                               mode='softmax',
                               sim_mode='dot_product'):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, T, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, T, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:

    """
    assert mode in ['softmax', 'cosine']
    batches = query.size(0)
    assert query.size(0) == key.size(0) == value.size(0)
    assert value.shape[2:] == key.shape[2:], f'{value.shape} {key.shape}'
    if key.ndim == 4:
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
    assert value.ndim == key.ndim == 5
    clip_len = key.size(2)
    assert 0 <= non_mask_len < clip_len
    # assert query.shape[2:] == key.shape[3:]
    att_channels, query_height, query_width = query.shape[1:]
    key_height, key_width = key.shape[3:]
    C = value.size(1)
    if normalize:
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)
    query_vec = query.view(batches, att_channels, query.shape[2:].numel())
    key_vec = key.view(batches, att_channels, key.shape[2:].numel())
    value_vec = value.view(batches, C, value.shape[2:].numel())
    output = torch.zeros(batches, C,
                         query_height * query_width).to(query)
    if step is None:
        step = query_height * query_width
    for ptr in range(0, query_height * query_width, step):
        # [N, TxHxW, step]
        if sim_mode == 'dot_product':
            cur_affinity = torch.einsum('bci,bcj->bij', key_vec,
                                        query_vec[...,
                                                ptr:ptr + step]) / temperature
        elif sim_mode == 'l2-distance':
            a_sq = key_vec.pow(2).sum(1).unsqueeze(2)
            ab = key_vec.transpose(1, 2) @ query_vec[...,ptr:ptr + step]
            cur_affinity = (2*ab-a_sq) / math.sqrt(att_channels)

        if mask is not None:
            if mask.ndim == 2:
                assert mask.shape == (key_height * key_width,
                                      query_height * query_width)
                cur_mask = mask.view(1, 1, key_height * key_width,
                                     query_height *
                                     query_width)[..., ptr:ptr + step].expand(
                                         batches, clip_len - non_mask_len, -1,
                                         -1).reshape(batches, -1,
                                                     cur_affinity.size(2))
            else:
                assert clip_len == 1
                assert non_mask_len == 0
                cur_mask = mask[..., ptr:ptr + step]

            if non_mask_len > 0:
                cur_mask = cat([
                    torch.ones(batches, non_mask_len * key_height * key_width,
                               cur_affinity.size(2)).to(cur_mask), cur_mask
                ],
                               dim=1)
            cur_affinity.masked_fill_(~cur_mask.bool(), float('-inf'))
        if topk is not None:
            # [N, topk, step]
            topk_affinity, topk_indices = cur_affinity.topk(k=topk, dim=1)
            # cur_affinity, idx = cur_affinity.sort(descending=True, dim=1)
            # topk_affinity, topk_indices = cur_affinity[:, :topk], idx[:,
            # :topk]
            topk_value = value_vec.transpose(0, 1).reshape(
                C, -1).index_select(
                    dim=1, index=topk_indices.reshape(-1))
            # [N, C, topk, step]
            topk_value = topk_value.reshape(C,
                                            *topk_indices.shape).transpose(
                                                0, 1)
            if mode == 'softmax':
                topk_affinity = topk_affinity.softmax(dim=1)
            elif mode == 'cosine':
                topk_affinity = topk_affinity.clamp(min=0)**2
            else:
                raise ValueError
            cur_output = torch.einsum('bcks,bks->bcs', topk_value,
                                      topk_affinity)
        else:
            if mode == 'softmax':
                cur_affinity = cur_affinity.softmax(dim=1)
            elif mode == 'cosine':
                cur_affinity = cur_affinity.clamp(min=0)**2
            else:
                raise ValueError
            cur_output = torch.einsum('bck,bks->bcs', value_vec, cur_affinity)
        output[..., ptr:ptr + step] = cur_output

    output = output.reshape(batches, C, query_height,
                            query_width)

    return output


def masked_attention_efficient_v2(query,
                               key,
                               value,
                               radius,
                               temperature=1,
                               topk=None,
                               normalize=True,
                               non_mask_len=0,
                               mode='softmax'):
    """

    Args:
        v1 is only memory efficient.

        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, T, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, T, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:

    """
    from mmcv.ops import Correlation

    corr = Correlation(max_displacement=radius)

    assert mode in ['softmax', 'cosine']
    batches = query.size(0)
    assert query.size(0) == key.size(0) == value.size(0)
    assert value.shape[2:] == key.shape[2:], f'{value.shape} {key.shape}'
    if key.ndim == 4:
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
    assert value.ndim == key.ndim == 5
    clip_len = key.size(2)
    assert 0 <= non_mask_len < clip_len
    # assert query.shape[2:] == key.shape[3:]
    att_channels, query_height, query_width = query.shape[1:]
    key_height, key_width = key.shape[3:]
    C = value.size(1)
    if normalize:
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)

    output = torch.zeros(batches, C,
                         query_height * query_width).to(query)
    query_vec = query.repeat(clip_len, 1, 1, 1)
    key_vec = key.transpose(1,2).flatten(0,1)

    # [NT, 3xR^2, H, W]
    value_vec = F.unfold(value.transpose(1,2).flatten(0,1), 2 * radius + 1, padding=radius)
    # [N, T, 3, R^2, HW]
    value_vec = value_vec.view(batches, clip_len, C, -1, key_height * key_width)
    # [N, 3, T*R^2, H, W]
    value_vec = value_vec.transpose(1,2).flatten(2,3)

    # [N, TxR^2, HW]
    affinity = corr(query_vec, key_vec).view(batches, -1, query_height * query_width) / temperature

    if topk is not None:
        # [N, topk, step]
        topk_affinity, topk_indices = affinity.topk(k=topk, dim=1)
        topk_value = value_vec.transpose(0, 1).reshape(
            C, -1).index_select(
                dim=1, index=topk_indices.reshape(-1))
        # [N, C, topk, step]
        topk_value = topk_value.reshape(C,
                                        *topk_indices.shape).transpose(
                                            0, 1)
        if mode == 'softmax':
            topk_affinity = topk_affinity.softmax(dim=1)
        elif mode == 'cosine':
            topk_affinity = topk_affinity.clamp(min=0)**2
        else:
            raise ValueError
        output = torch.einsum('bcks,bks->bcs', topk_value,
                                topk_affinity)
    else:
        if mode == 'softmax':
            affinity = affinity.softmax(dim=1)
        elif mode == 'cosine':
            affinity = affinity.clamp(min=0)**2
        else:
            raise ValueError
        output = torch.einsum('bck,bks->bcs', value_vec, affinity)


    output = output.reshape(batches, C, query_height,
                            query_width)

    return output



def flow_guided_attention_efficient(query,
                               key,
                               value,
                               h_feat,
                               cxt_feat,
                               sample_fn,
                               decoder,
                               topk=None,
                               step=32,
                               t_step=5,
                               radius=6,
                               temperature=0.07,
                               mode='softmax',
                               with_norm=True
                               ):
    """

    Args:
        query(torch.Tensor): Value tensor, shape (T, C, H, W)
        key (torch.Tensor): Value tensor, shape (T, C, H, W)
        value (torch.Tensor): Value tensor, shape (T, C, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:

    """
    L, C, H, W = value.shape
    C_ = query.shape[1]
    decoder = decoder.cuda()

    if step is None:
        step = H * W

    output = torch.zeros(1, C,
                         H * W).to(value)

    for ptr in range(0, H*W, step):
        s = min(H*W - ptr, step)
        affinity = torch.zeros(L, (radius*2 +1)**2,
                            s).to(value)
        value_ = []

        start = time.time()

        for ptr_t in range(0, L, t_step):
            
            t = min(L- ptr_t, t_step) 

            flow_init = torch.zeros((t, 2, *query.shape[-2:]), device=query.device)
            value_feat = value[ptr_t:ptr_t+t_step]

            pred, corr = decoder(
                    True,
                    query[ptr_t:ptr_t+t_step],
                    key[ptr_t:ptr_t+t_step],
                    flow=flow_init,
                    h_feat=h_feat[ptr_t:ptr_t+t_step],
                    cxt_feat=cxt_feat[ptr_t:ptr_t+t_step],
                )

            if not with_norm:
                pass
            else:
                query_feat = query[ptr_t:ptr_t+t_step].pow(2).sum(dim=1, keepdim=True).sqrt()
                key_feat = key[ptr_t:ptr_t+t_step].pow(2).sum(dim=1, keepdim=True).sqrt()
                # L'HW x 1 x H x W
                norm_ = torch.matmul(query_feat.flatten(-2).permute(0,2,1), key_feat.flatten(-2)).reshape(t*H*W, 1, H, W)
                corr = (corr * torch.sqrt(torch.tensor(C_).float())/ norm_) / temperature
                corr = corr.reshape(t, -1, 1, H, W)
            
            xx = torch.arange(0, W, device=pred.device)
            yy = torch.arange(0, H, device=pred.device)
            # L' x 2 x H x W
            grid = coords_grid(t, xx, yy) + pred
            grid = grid.permute(0, 2, 3, 1)
            grid = grid.flatten(1,2)

            # L' x S x 2
            g = grid[:, ptr:ptr + step]
            c = corr[:, ptr:ptr + step]
            # L'S x 1 x H x W
            c = c.flatten(0,1)

            # L' x r^2 x S
            cur_affinity = sample_fn([c], flow=None, grid=g)
            affinity[ptr_t:ptr_t+t_step, ...] = cur_affinity

            # for valune
            # L'S x C x H x W 
            v = value_feat.repeat(1, s, 1, 1, 1).flatten(0,1)
            # 1 x L' x C x r^2 x S
            ref_v = sample_fn([v], flow=None, mode='nearest', grid=g).reshape(t, C, -1, s).unsqueeze(0)
            value_.append(ref_v)

        if topk is not None:
            # 1 x (L' x r^2) x S
            affinity = affinity.reshape(1, -1, s)

            # [1, topk, S]
            topk_affinity, topk_indices = affinity.topk(k=topk, dim=1)
            
            # 1 x C x (L x r^2) x S
            value_ = torch.cat(value_, 1).transpose(1,2).flatten(2,3)

            topk_value = value_.transpose(0, 1).reshape(
                C, -1).index_select(
                    dim=1, index=topk_indices.reshape(-1))
            # [N, C, topk, step]
            topk_value = topk_value.reshape(C,
                                            *topk_indices.shape).transpose(
                                                0, 1)
            if mode == 'softmax':
                topk_affinity = topk_affinity.softmax(dim=1)
            elif mode == 'cosine':
                topk_affinity = topk_affinity.clamp(min=0)**2
            else:
                raise ValueError

            cur_output = torch.einsum('bcks,bks->bcs', topk_value,
                                    topk_affinity)
        # print(time.time()- start)
        output[...,ptr:ptr+step] = 0
        
    return output

def flow_guided_attention_efficient_v2(corr,
                                value,
                                pred,
                                sample_fn,
                               topk=10,
                               step=32,
                               mode='softmax',
                               ):
    """

    Args:
        query(torch.Tensor): Value tensor, shape (T, C, H, W)
        key (torch.Tensor): Value tensor, shape (T, C, H, W)
        value (torch.Tensor): Value tensor, shape (T, C, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:

    """
    L, C, H, W = value.shape

    if step is None:
        step = H * W

    output = torch.zeros(1, C,
                         H * W).to(value)

    xx = torch.arange(0, W, device=pred.device)
    yy = torch.arange(0, H, device=pred.device)
    # L x 2 x H x W
    grid = coords_grid(L, xx, yy) + pred
    grid = grid.permute(0, 2, 3, 1)
    grid = grid.flatten(1,2)
    corr = corr.reshape(L, H*W, 1, H, W)

    for ptr in range(0, H*W, step):
        s = min(H*W - ptr, step)

        # L x S x 2
        g = grid[:, ptr:ptr + step]
        c = corr[:, ptr:ptr + step]

        # LS x 1 x H x W
        c = c.flatten(0,1)

        # L' x r^2 x S
        cur_affinity = sample_fn([c], flow=None, grid=g)

        if topk is not None:
            # 1 x (L x r^2) x S
            cur_affinity = cur_affinity.reshape(1, -1, s)

            # [1, topk, S]
            topk_affinity, topk_indices = cur_affinity.topk(k=topk, dim=1)
            
            v = value.repeat(1, s, 1, 1, 1).flatten(0,1)
            # 1 x L x C x r^2 x S
            value_ = sample_fn([v], flow=None, mode='nearest', grid=g).reshape(L, C, -1, s).unsqueeze(0)
            # 1 x C x (L x r^2) x S
            value_ = value_.transpose(1,2).flatten(2,3)

            topk_value = value_.transpose(0, 1).reshape(
                C, -1).index_select(
                    dim=1, index=topk_indices.reshape(-1))
            # [N, C, topk, step]
            topk_value = topk_value.reshape(C,
                                            *topk_indices.shape).transpose(
                                                0, 1)
            if mode == 'softmax':
                topk_affinity = topk_affinity.softmax(dim=1)
            elif mode == 'cosine':
                topk_affinity = topk_affinity.clamp(min=0)**2
            else:
                raise ValueError

            cur_output = torch.einsum('bcks,bks->bcs', topk_value,
                                    topk_affinity)

        output[...,ptr:ptr+step] = cur_output

    return output


if __name__ == '__main__':
    import time
    s = 128
    b = 1

    q = torch.rand((b, 256, s, s)).cuda()
    k = torch.rand((b, 256, 20, s, s)).cuda()
    v = torch.rand((b, 3, 20, s, s)).cuda()
    mask = torch.ones((s*s,s*s)).cuda()

    start = time.time()
    # _ = masked_attention_efficient(q, k, v, mask, topk=10, step=32)
    _ = masked_attention_efficient_v2(q, k, v, topk=10, radius=24)


    print(time.time()-start)