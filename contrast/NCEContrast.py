import torch
import math
from torch import nn


class MemorySTM(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, feature_dim, queue_size, temperature=0.07):
        super(MemorySTM, self).__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.index = 0
        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)
        self.register_buffer('queue_label', torch.ones(self.queue_size, dtype=torch.long) * -1)
        
    def forward(self, q_feat, q_region, k_feat, labels, k_feat_all, k_label_all):
        try:
            l_pos_sf = (q_feat.unsqueeze(1) * k_feat.detach()).sum(dim=-1, keepdim=True).reshape(16, 1)  # shape: (batchSize, 20, 1)
        except Exception as e:
            print('haha')
        l_neg = torch.mm(q_feat, self.memory.clone().detach().t())

        l_pos_sf_region = (q_region.unsqueeze(1) * k_feat.detach()).sum(dim=-1, keepdim=True).reshape(16, 1)  # shape: (batchSize, 20, 1)
        l_neg_region = torch.mm(q_region, self.memory.clone().detach().t())
  
        out = torch.cat((l_pos_sf, l_neg.repeat(16, 1)), dim=1)
        out = torch.div(out, self.temperature).contiguous()

        out_region = torch.cat((l_pos_sf, l_neg.repeat(16, 1)), dim=1)
        out_region = torch.div(out, self.temperature).contiguous()

        # filter self
        mask_source = labels.unsqueeze(1) == self.queue_label.unsqueeze(0)
        mask_source = mask_source.type(torch.float32)
        mask = torch.cat((torch.ones(mask_source.size(0), 1).cuda(), mask_source), dim=1).cuda()

        with torch.no_grad(): # update memory
            k_all = k_feat_all
            all_size = k_all.shape[0]
            out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
            self.memory.index_copy_(0, out_ids, k_all)
            self.queue_label.index_copy_(0, out_ids, k_label_all)
            self.index = (self.index + all_size) % self.queue_size
        return out, out_region, mask, l_pos_sf
    

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        label = torch.zeros([x.shape[0]]).long().to(x.device)
        return self.criterion(x, label)
