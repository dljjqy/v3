# import torch
import torchvision.transforms as transforms
# import random
import numpy as np
import re
import random
from collections import namedtuple

Pair = namedtuple('Pair', ['left', 'right', 'l_disp'])

NORM_KWARGS = {'mean':[0.485, 0.456, 0.406],'std':[0.229, 0.224, 0.225]}
class ToDisp:

    def __init__(self, crop_size = (256, 512)):
        self.crop_size = crop_size
    
    def __call__(self, img):
        # convert image from uint16 to uint8 based on the developkit 
        if self.crop_size :
            img = transforms.RandomCrop(self.crop_size)(img)

        arr = np.ascontiguousarray(img, dtype=np.float)/256
        tensor = transforms.ToTensor()(arr)
        return tensor

class Rdomcrop_ToTensor_Norm:
    def __init__(self, crop_size = (256, 512)):
        self.crop_size = crop_size
        self.transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(**NORM_KWARGS)
        ])
    
    def __call__(self, image):
        return self.transform(image)

class ToTenso_Norm:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**NORM_KWARGS)
        ])
    def __call__(self, image):
        return self.transform(image)


class RandomCrop:
    def __init__(self, crop_size=(256, 512)):
        self.crop_size = crop_size
    def __call__(self, pair):
        w, h = pair.left.size
        th, tw = self.crop_size

        x = random.randint(0, w - tw)
        y = random.randint(0, h - th)

        left_img = pair.left.crop((x, y, x + tw, y + th))
        right_img = pair.right.crop((x, y, x + tw, y + th))
        disp = pair.l_disp[y:y+th, x:x+tw]
        return Pair(left_img, right_img, disp)

class RandCrop_ToTensor_Norm_for_pair:
    def __init__(self, crop_size=(256, 512)):
        self.crop_size = crop_size
        self.randcrop = RandomCrop(self.crop_size)
        self.transform = ToTenso_Norm()  

    def __call__(self, pair):
        left, right, disp = self.randcrop(pair)
        disp = transforms.ToTensor()(disp)
        left, right = self.transform(left), self.transform(right)
        return Pair(left, right, disp)

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (3, height, width) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    data = np.ascontiguousarray(data, dtype=np.float32)
    file.close()
    return data, scale

if __name__ == "__main__":
    test_pfm_path = '/media/zxpwhu/zxp/datasets/SceneFlow/flyingthings3d/disparity/TRAIN/A/0000/left/0006.pfm'
    data, scale = readPFM(test_pfm_path)
    print(type(data), type(scale))
    print(data.dtype)
    print(data.shape)
    test = np.ascontiguousarray(data, dtype=np.float32)
    print(test-data)
    # H x W