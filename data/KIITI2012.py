import torch
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image 
from torch.utils.data import Dataset
from collections import namedtuple
from .helper import *

class KITTI2012_dataset(Dataset):

    def __init__(self, root, do_test=False):
        super(KITTI2012_dataset, self).__init__()
        self.root = Path(root)
        self.do_test = do_test
        self._correspond()

    def __getitem__(self, index):
        assert index < self.__len__(), 'Index Error'

        img_name = self.img_names[index]
        left_img = Image.open(self.left_root/img_name)
        # print('left', (self.left_root/img_name).name)

        right_img = Image.open(self.right_root/img_name)
        left, right = Rdomcrop_ToTensor_Norm()(left_img), Rdomcrop_ToTensor_Norm()(right_img)
        
        if not self.do_test:
            disp = Image.open(self.disp_root/img_name)

            # print('disp', (self.disp_root/img_name).name)
            disp = ToDisp()(disp)        
            return Pair(left, right, disp)
        else:
            return Pair(left, right, None)


    def __len__(self):
        return len(self.img_names)

    def _correspond(self):

        if self.do_test:
            self.root = self.root/'testing'
            self.left_root = self.root/'colored_0'
            self.right_root = self.root/'colored_1'
            self.img_names = [img.name for img in self.left_root.glob('*.png')]

        else:
            self.root = self.root/'training'
            self.left_root = self.root/'colored_0'
            self.right_root = self.root/'colored_1'
            self.disp_root = self.root/'disp_occ'
            self.img_names = [img.name for img in self.disp_root.glob('*png')]

if __name__ == "__main__":
    
    root = '/media/zxpwhu/zxp/datasets/KITTI/stereo2012/data_stereo_flow2012'
    dataset = KITTI2012_dataset(root)
    dataiter = iter(dataset)
    l = []
    for i in range(5):
        l.append(dataiter.__next__())
    
    left, right, disp = l[3]
    print(f'left shape:{left.shape}, right shape:{right.shape}, disp shape:{disp.shape}')