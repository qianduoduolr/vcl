import os
import numpy as np
import cv2
import torch

import mmcv
from PIL import Image
import copy
import matplotlib.pyplot as plt
from vcl.models.common.correlation import *



def visualize_att(imgs, atts, idx, abs_att=False, query_idx=None, feat_size=25, patch_size=5, dst_path='/home/lr/project/vcl/output/vis', norm_mode='mean-std', color='bone'):
    bsz, num_clips, t, c, w, h = imgs.shape

    dst_path = os.path.join(dst_path, f'{idx}')
    mmcv.mkdir_or_exist(dst_path)


    atts = list([att.cpu().numpy() for att in atts])
    atts = np.stack(atts, 2)
    atts = atts[0, query_idx.item()]
    if abs_att:
        atts = ((atts - atts.min()) * 255 / (atts.max() - atts.min())).astype(np.uint8)
    else:
        atts = ((atts - atts.min(-1)) * 255 / (atts.max(-1) - atts.min(-1))).astype(np.uint8)
        

    outs = []
    for i in range(t):
        img = imgs[:,:,i].reshape(c,w,h)
        out = tensor2img(img, norm_mode=norm_mode)
        outs.append(out)

    blend_outs = []
    for index, i in enumerate(range(atts.shape[0])):
        
        if patch_size == -1:
            attr = atts[i].reshape(feat_size, feat_size)
            # attr = (att[:, query_idx.item()]).reshape(feat_size, feat_size)
            att_out = attr
            att_out_query = np.zeros(feat_size * feat_size).astype(np.uint8)
            att_out_query[query_idx.item()] = 255
            att_out_query = att_out_query.reshape(feat_size, feat_size)
        else:
            att = att.cpu().detach().numpy()
            x = query_idx.item() // feat_size
            y = query_idx.item() % feat_size

            att_out = np.zeros((feat_size, feat_size))
            attr = (att[:, query_idx.item()]).reshape(feat_size, feat_size)

            att_out[max(x-patch_size // 2,0):min(x+patch_size // 2+1,feat_size), max(y-patch_size // 2, 0):min(y+patch_size // 2+1, feat_size)] = attr[max(x-patch_size // 2,0):min(x+patch_size // 2+1,feat_size), max(y-patch_size // 2, 0):min(y+patch_size // 2+1, feat_size)]

            att_out = ((att_out - att_out.min()) * 255 / (att_out.max() - att_out.min())).astype(np.uint8)
            attr = ((attr - attr.min()) * 255 / (attr.max() - attr.min())).astype(np.uint8)

            att_out_query = np.zeros(feat_size * feat_size).astype(np.uint8)
            att_out_query[query_idx.item()] = 255
            att_out_query = att_out_query.reshape(feat_size, feat_size)


        
        resized_ = cv2.resize(att_out, (w,h))
        resized_query = cv2.resize(att_out_query, (w,h), cv2.INTER_NEAREST)


        att_map = cv2.applyColorMap(att_out, cv2.COLORMAP_BONE)
        attr = cv2.applyColorMap(attr, cv2.COLORMAP_BONE)
        att_query_map = cv2.applyColorMap(att_out_query, cv2.COLORMAP_BONE)

        if index == 0:
            img_ = Image.fromarray(copy.deepcopy(outs[index])).convert('RGBA')
            resized_out = Image.fromarray(resized_query).convert('RGBA')
            blend_out_query = Image.blend(img_, resized_out, 0.7)
            blend_out_query = np.array(blend_out_query)
            blend_out_query = np.concatenate([blend_out_query, np.ones((h,7,c+1))*255], 1)
            blend_outs.append(np.array(blend_out_query))
            
        img_ = Image.fromarray(copy.deepcopy(outs[index+1])).convert('RGBA')
        resized_ = Image.fromarray(resized_).convert('RGBA')
        blend_out = Image.blend(img_, resized_, 0.7)
        blend_out = np.array(blend_out)
        blend_out = np.concatenate([blend_out, np.ones((h,7,c+1))*255], 1)
        blend_outs.append(np.array(blend_out))
    
    blend_outs = np.concatenate(blend_outs, 1)
    outs = np.concatenate(outs, 1)
    
    
    cv2.imwrite(os.path.join(dst_path,'out_blend.png'), blend_outs)
    cv2.imwrite(os.path.join(dst_path,'out_img.png'), outs)
    
    

def visualize_vqvae(z1_q, z2_q, frame1, frame2, x_rec1, x_rec2, nembed=2048, rescale=True, mask=None):
    frame_vq_1 = z1_q.permute(1,2,0).numpy()
    frame_vq_2 = z2_q.permute(1,2,0).numpy()

    print(len(np.unique(frame_vq_1)))

    if rescale:
        frame_vq_1 = (frame_vq_1 * 255 / nembed).astype(np.uint8)
        frame_vq_2 = (frame_vq_2 * 255 / nembed).astype(np.uint8)
    else:
        frame_vq_1 = (frame_vq_1).astype(np.uint8)
        frame_vq_2 = (frame_vq_2).astype(np.uint8)

    if mask is not None:

        frame_vq_2 = frame_vq_2 * mask

    plt.rcParams['figure.dpi'] = 200

    plt.figure()

    if x_rec1 is not None:
        plt.subplot(3,2,1)
        plt.imshow(frame_vq_1, cmap=plt.get_cmap('jet'))
        plt.subplot(3,2,2)
        plt.imshow(frame_vq_2, cmap=plt.get_cmap('jet'))

        plt.subplot(3,2,3)
        plt.imshow(np.array(frame1))

        plt.subplot(3,2,4)
        plt.imshow(np.array(frame2))

        plt.subplot(3,2,5)
        plt.imshow(np.array(x_rec1))

        plt.subplot(3,2,6)
        plt.imshow(np.array(x_rec2))

        plt.show()
    else:
        plt.subplot(2,2,1)
        plt.imshow(frame_vq_1, cmap=plt.get_cmap('jet'))
        plt.subplot(2,2,2)
        plt.imshow(frame_vq_2, cmap=plt.get_cmap('jet'))

        plt.subplot(2,2,3)
        plt.imshow(np.array(frame1))

        plt.subplot(2,2,4)
        plt.imshow(np.array(frame2))


    
def visualize_correspondence_quant(z1_q, z2_q, sample_idx, frame1, frame2, scale=32):
    plt.rcParams['figure.dpi'] = 200
    
    x, y = sample_idx % scale, sample_idx // scale

    querys_map = np.zeros((scale,scale))
    querys_map[y,x] = 255
    querys_map = querys_map.astype(np.uint8)
    
    _, att = non_local_attention(z1_q, [z2_q], flatten=False)
    
    # print(att[0,0, sample_idx].max(), att[0,0,sample_idx].sum())
    
    att = att[0, 0, sample_idx].reshape(scale,scale).detach().cpu().numpy() * 255
    
    # print(att.max())
    att = ((att - att.min()) * 255 / (att.max() - att.min())).astype(np.uint8)
    
    att = cv2.resize(att, (256,256))
    querys_map = cv2.resize(querys_map, (256,256))
    
    att_map = cv2.applyColorMap(att.astype(np.uint8), cv2.COLORMAP_BONE)
    att_query_map = cv2.applyColorMap(querys_map, cv2.COLORMAP_BONE)


    tar = Image.fromarray(frame1).convert('RGBA')
    query = Image.fromarray(att_query_map).convert('RGBA')
    blend_out_query = Image.blend(tar, query, 0.7)
    blend_out_query = np.array(blend_out_query)

    ref = Image.fromarray(frame2).convert('RGBA')
    result = Image.fromarray(att_map).convert('RGBA')
    blend_out_result = Image.blend(ref, result, 0.7)
    blend_out_result = np.array(blend_out_result)


    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(blend_out_query)
    plt.subplot(2,2,2)
    plt.imshow(blend_out_result)
    plt.subplot(2,2,3)
    plt.imshow(querys_map)
    plt.subplot(2,2,4)
    plt.imshow(att)
    
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(querys_map)
    # plt.subplot(1,2,2)
    # plt.imshow(att)

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


