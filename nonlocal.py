import torch
from torch import nn
from torch.nn import functional as F
import ipdb


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels,inter_channels, dimension=2, sub_sample=False, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()



        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
	if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        #self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
         #                    kernel_size=1, stride=1, padding=0)
        #self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                   kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)


        
	#theta_x = self.theta(x).view(batch_size,  -1)
        
        #phi_x = self.phi(x).view(batch_size,  -1)
	theta_x =x.view(batch_size,  -1)
        
        phi_x = x.view(batch_size,  -1)
	phi_x = theta_x.permute(1,0)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=0)



	return f_div_C

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels, sub_sample=False, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,inter_channels=inter_channels,dimension=2, sub_sample=sub_sample,bn_layer=bn_layer)






