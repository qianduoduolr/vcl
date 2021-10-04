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
import numbers
import os.path as osp
from collections import *

import mmcv
from mmcv.runner import auto_fp16

from ..base import BaseModel
from ..builder import build_backbone, build_loss, bulid_component
from ..registry import MODELS

@MODELS.register_module()
class VFS(BaseModel):
    def __init__(self, depth, 
                       nce_loss, 
                       memory_bank, 
                       train_cfg=None, 
                       test_cfg=None, 
                       pretrained=None):
        """[summary]

        Args:
            self ([type]): [description]
            contrast_loss ([type]): [description]
            train_cfg ([type], optional): [description]. Defaults to None.
            test_cfg ([type], optional): [description]. Defaults to None.
            pretrained ([type], optional): [description]. Defaults to None.
        """
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.encoder_q = bulid_backbone(dict(type='ResNet', depth=depth))
        self.encoder_k = bulid_backbone(dict(type='ResNet', depth=depth))

        self.nce_loss = bulid_loss(nce_loss)

        self.memory_bank = bulid_component(memory_bank)

    
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

    def forward_train(self, Fs, Ms, num_objects):
        pass
