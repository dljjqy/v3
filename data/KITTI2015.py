import torch
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image 
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from .helper import *

class KITTI2015_dataset(Dataset):

    def __init__(self, root, do_test=False):
        super(KITTI2015_dataset, self).__init__()
        self.root = Path(root)
        self.do_test = do_test
        self._correspond()

    def __getitem__(self, index):
        assert index < self.__len__(), 'Index Error'

        img_name = self.img_names[index]
        left_img = Image.open(self.left_root/img_name)
        right_img = Image.open(self.right_root/img_name)

        print('left', (self.left_root/img_name).name)

        if self.do_test:
            left, right = Rdomcrop_ToTensor_Norm()(left_img), Rdomcrop_ToTensor_Norm()(right_img)
            return Pair(left, right, None)

        else:
            left, right = Rdomcrop_ToTensor_Norm()(left_img), Rdomcrop_ToTensor_Norm()(right_img)
            # Get corresponding disparity image
            info = img_name.split('_')
            img_id, img_is_first = info[0], True if '10' in info[1] else False
            
            if img_is_first:
                disp_name = self.disp_root_0/(img_id+'_10.png')
                disp = Image.open(disp_name)
            else:
                disp_name = self.disp_root_1/(img_id+'_10.png')
                disp = Image.open(disp_name)
            
            disp = ToDisp()(disp)
            return Pair(left, right, disp)

    def __len__(self):
        return len(self.img_names)

    def _correspond(self):

        if self.do_test:
            self.root = self.root/'testing'
            self.left_root = self.root/'image_2'
            self.right_root = self.root/'image_3'
            self.img_names = [img.name for img in self.left_root.glob('*.png')]

        else:
            self.root = self.root/'training'
            self.left_root = self.root/'image_2'
            self.right_root = self.root/'image_3'
            self.disp_root_0 = self.root/'disp_occ_0'
            self.disp_root_1 = self.root/'disp_occ_1'

            self.img_names = [img.name for img in self.left_root.glob('*png')]

# KITTI2015_dataloader = DataLoader(**loader_kwargs)
if __name__ == "__main__":
    
    root = '/media/zxpwhu/zxp/datasets/KITTI/stereo2015/data_scene_flow2015'
    dataset = KITTI2015_dataset(root)
    dataiter = iter(dataset)
    l = []
    for i in range(5):
        l.append(dataiter.__next__())
    
    left, right, disp = l[3]
    print(f'left shape:{left.shape}, right shape:{right.shape}, disp shape:{disp.shape}')