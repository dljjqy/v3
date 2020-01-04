from submodule import *
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PSMNet(nn.Module):

    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp

        # CNN and SPP
        self.feature_extractor = feature_extraction()

        # Depth Estimation
        self.conv3d_0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True))

        self.conv3d_1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn_3d(32, 32, 3, 1, 1))

        self.conv3d_2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn_3d(32, 32, 3, 1, 1))

        self.conv3d_3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn_3d(32, 32, 3, 1, 1))

        self.conv3d_4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn_3d(32, 32, 3, 1, 1))

        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # initilize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * \
                    m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        ref = self.feature_extractor(left)
        tar = self.feature_extractor(right)

        # generating cost volumn
        # size = Batch X (Channels*2) X Disp X H' X W'
        vol_size = (ref.shape[0], (ref.shape[1]*2),
                    self.maxdisp//4, *ref.shape[2:])
                    
        cost = Variable(torch.FloatTensor(*vol_size).zero_(),
                        volatile=not self.training)
        print(f'###################cost shape: {cost.shape}')
        print(f'###################ref shape: {ref.shape}')
        print(f'###################tar shape: {tar.shape}')

        for i in range(self.maxdisp//4):
            if i > 0:
                cost[:, :ref.shape[1], i, :, i:] = ref[:, :, :, i:]
                cost[:, ref.shape[1]:, i, :, i:] = tar[:, :, :, :-i]
            else:
                cost[:, :ref.shape[1], i, :, :] = ref
                cost[:, ref.shape[1]:, i, :, :] = tar
        cost = cost.contiguous()

        cost0 = self.conv3d_0(cost)
        cost0 = self.conv3d_1(cost0) + cost0
        cost0 = self.conv3d_2(cost0) + cost0
        cost0 = self.conv3d_3(cost0) + cost0
        cost0 = self.conv3d_4(cost0) + cost0

        cost = self.classify(cost0)
        cost = F.interpolate(cost, [
                             self.maxdisp, left.shape[2], left.shape[3]], mode='trilinear', align_corners=True)
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost)
        pred = disparityregression(self.maxdisp)(pred)
        return pred


if __name__ == "__main__":
    import torch
    left = torch.rand(1, 3, 256, 512)
    right = torch.rand(1, 3, 256, 512)

    net = PSMNet(124)
    output = net(left, right)
