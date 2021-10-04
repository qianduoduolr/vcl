from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.core.evaluation.metrics import JFM
from .stm import *

@MODELS.register_module()
class STM_SCL(STM):
    def __init__(self, depth, pixel_loss, contrast_loss, train_cfg=None, test_cfg=None, pretrained=None):
        """[summary]

        Args:
            depth ([type]): [description]
            pixel_loss ([type]): [description]
            train_cfg ([type], optional): [description]. Defaults to None.
            test_cfg ([type], optional): [description]. Defaults to None.
            pretrained ([type], optional): [description]. Defaults to None.
        """

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        scale_rate = (1 if (depth == 50 or depth == 101) else 4)

        self.Encoder_M = Encoder_M(depth) 
        self.Encoder_Q = Encoder_Q(depth) 
        self.Encoder_Q_M = Encoder_Q(depth) 

        self.KV_M_r4 = KeyValue(1024//scale_rate, keydim=128//scale_rate, valdim=512//scale_rate)
        self.KV_Q_r4 = KeyValue(1024//scale_rate, keydim=128//scale_rate, valdim=512//scale_rate)

        self.proj = nn.Sequential(
            nn.Conv2d(1024//scale_rate,1024//scale_rate, 1),
            nn.ReLU(),
            nn.Conv2d(1024//scale_rate,128, 1)
        )

        self.proj_m = nn.Sequential(
            nn.Conv2d(1024//scale_rate,1024//scale_rate, 1),
            nn.ReLU(),
            nn.Conv2d(1024//scale_rate,128, 1)
        )

        self.Memory = Memory()
        self.Decoder = Decoder(256,scale_rate)

        # loss
        self.pixel_ce_loss = build_loss(pixel_loss)
        self.cts_loss = build_loss(contrast_loss)

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

        r4k, _, _, _, _ = self.Encoder_Q(B_['f'])
        r4v, _, _, _, _ = self.Encoder_M(B_['f'], B_['m'], B_['o'])

        k4 = self.KV_Q_r4(r4k, mode='key') # num_objects, 128 and 512, H/16, W/16
        v4 = self.KV_M_r4(r4v, mode='value')
        k4, v4 = self.Pad_memory([k4, v4], num_objects=num_objects, K=K)

        return k4, v4
    
    def segment(self, frame, keys, values, num_objects, contrast=False): 
        num_objects = num_objects[0].item()

        _, K, keydim, T, H, W = keys.shape # B = 1
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4  = self.KV_Q_r4(r4, mode='key')   # 1, dim, H/16, W/16
        v4  = self.KV_Q_r4(r4, mode='value')   # 1, dim, H/16, W/16
    
        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1) 
        r3e, r2e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1)
        
        # memory select kv:(1, K, C, T, H, W)
        m4, viz = self.Memory(keys[0,1:num_objects+1], values[0,1:num_objects+1], k4e, v4e)
        if self.backbone == 'resnest101':
            m4 = self.aspp(m4)
        logits = self.Decoder(m4, r3e, r2e)
        ps = F.softmax(logits, dim=1)[:,1] # no, h, w  
        #ps = indipendant possibility to belong to each object
        
        logit = self.Soft_aggregation(ps, K) # 1, K, H, W

        if pad[2]+pad[3] > 0:
            logit = logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            logit = logit[:,:,:,pad[0]:-pad[1]]

        if contrast:
            k4c = self.proj(r4)
            return logit, k4c
        else:
            return logit

    def cts(self, x):
        r4k, _, _, _, _ = self.Encoder_Q_M(x)
        k4c = self.proj_m(r4k)
        return k4c

    def forward_train(self, Fs, Ms, num_objects, Ms_obj=None):
        """[summary]

        Args:
            Fs ([type]): [description]
            Ms ([type]): [description]
            num_objects ([type]): [description]
            Ms_obj ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
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
