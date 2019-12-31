import torch
import numpy as np
import torch.optim as optim
from data import *
from torch.utils.data import DataLoader
# from utils.lr_scheduler import const_lr
import torch.nn.functional as F
from core.models.basic import PSMNet
from core.trainer import Trainer
# Dataasets
kitti2015_dataset = KITTI2015_dataset('/media/zxpwhu/zxp/datasets/KITTI/stereo2015/data_scene_flow2015')
kitti2012_dataset = KITTI2012_dataset('/media/zxpwhu/zxp/datasets/KITTI/stereo2012/data_stereo_flow2012')
sceneflow_dataset = SceneFlow('/media/zxpwhu/zxp/datasets/SceneFlow/')

# Loaders
# Train on sceneflow dataset
train_loader_kwargs = {
    'dataset':sceneflow_dataset,
    'batch_size':8,
    'shuffle':True,
    'num_workers':2,
    'drop_last':False,
    'pin_memory':True,
}
train_loader = DataLoader(**train_loader_kwargs)

test_loader_kwargs = {
    'dataset':kitti2015_dataset,
    'batch_size':8,
}
test_loader = DataLoader(**test_loader_kwargs)

Loss = F.smooth_l1_loss
net = PSMNet(192)
optimzier = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

device = torch.device('cuda')
net = net.to(device)
trainer_kwargs = {
    'device':device,
    'epochs':100,
    'dataloader':train_loader,
    'net':net,
    'optimizer':optimzier,
    'lr_scheduler':None,
    'loss':Loss,
}
trainer = Trainer(**trainer_kwargs)
if __name__ == "__main__":
    trainer.train()