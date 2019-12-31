import torch
import numpy as np
from PIL import Image
from .helper import readPFM
from torch.utils.data import Dataset
from pathlib import Path
from collections import namedtuple
from .helper import *


class SceneFlow(Dataset):
    def __init__(self, root):
        super(SceneFlow, self).__init__()
        self.root = Path(root)
        self._correspond()

    
    def _correspond(self):
        self.files = []

        driving = self.root/'driving'
        driving_list = []
        flyingthings3d = self.root/'flyingthings3d'
        flyingthings3d_list = []
        # Flyingthings3D_subset = self.root/'Flyingthins3D_subset'
        # Flyingthings3D_subset_list = []
        monkaa = self.root/'monkaa'
        monkaa_list = []

        driving_disp = driving/'disparity'
        driving_img = driving/'frames_cleanpass'
        for sub0 in driving_disp.iterdir():
            for sub1 in sub0.iterdir():
                for sub2 in sub1.iterdir():
                    img_path = driving_img/sub0.name/sub1.name/sub2.name
                    left_img_path = img_path/'left'
                    right_img_path = img_path/'right'
                    left_disp_path = sub2/'left'
                    for l_disp in left_disp_path.iterdir():
                        driving_list.append(Pair(left_img_path/(l_disp.name.split('.')[0]+'.png'), 
                                            right_img_path/(l_disp.name.split('.')[0]+'.png'), 
                                            l_disp))
                        
        flythings3d_disp = flyingthings3d/'disparity'/'TRAIN'
        flythings3d_img = flyingthings3d/'frames_cleanpass'/'TRAIN'
        for sub0 in flythings3d_disp.iterdir(): # A, B, C
            for sub1 in sub0.iterdir(): # nbr
                left_disp_path = sub1/'left'
                left_img_path = flythings3d_img/sub0.name/sub1.name/'left'
                right_img_path = flythings3d_img/sub0.name/sub1.name/'right'
                for l_disp in left_disp_path.iterdir():
                        flyingthings3d_list.append(Pair(left_img_path/(l_disp.name.split('.')[0]+'.png'), 
                                            right_img_path/(l_disp.name.split('.')[0]+'.png'), 
                                            l_disp))

        monkaa_disp = monkaa/'disparity'
        monkaa_img = monkaa/'frames_cleanpass'
        for sub0 in monkaa_disp.iterdir():
            left_disp_path = sub0/'left'
            left_img_path = monkaa_img/sub0.name/'left'
            right_img_path = monkaa_img/sub0.name/'right'
            for l_disp in left_disp_path.iterdir():
                        monkaa_list.append(Pair(left_img_path/(l_disp.name.split('.')[0]+'.png'), 
                                            right_img_path/(l_disp.name.split('.')[0]+'.png'), 
                                            l_disp))
                

        self.files = [*driving_list,*flyingthings3d_list,*monkaa_list  ]
        pass

    def __getitem__(self, index):
        assert index < self.__len__(), 'Index Error'
        left_path, right_path, pfm_path = self.files[index]
        left_img = Image.open(left_path)
        right_img = Image.open(right_path)
        left_disp, scale = readPFM(pfm_path)
        
        pair = Pair(left_img, right_img, left_disp)
        

        return RandCrop_ToTensor_Norm_for_pair()(pair)

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    root = '/media/zxpwhu/zxp/datasets/SceneFlow/'
    dataset = sceneflow(root)
    dataiter = iter(dataset)
    l = [dataiter.__next__() for i in range(10)]
    for i in l:
        for j in i:
            print(f'{j.shape}, ')
        print('\n ####################################')

    