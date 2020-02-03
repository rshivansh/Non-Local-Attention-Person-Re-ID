import os.path as osp

# import fcn							# changed this
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import ipdb
from nonlocal import NONLocalBlock2D

class FCN32s(nn.Module):

    def __init__(self, n_class=1):
        super(FCN32s, self).__init__()

        padDim = 4
	nFilters = [16,32,32]
	filtsize = [5,5,5]
    	poolsize = [2,2,2]
    	stepSize = [2,2,2]

	ninputChannels = 5

 	self.padding_1 = nn.ZeroPad2d(padDim)
	self.cnn_conv1 = nn.Conv2d(ninputChannels, nFilters[0], (filtsize[0], filtsize[0]), (1, 1))
	self.tanh1 = nn.Tanh()
	self.maxpool1 = nn.MaxPool2d((poolsize[0],poolsize[0]),(stepSize[0],stepSize[0])) 

	ninputChannels = nFilters[0]
	self.padding_2 = nn.ZeroPad2d(padDim)
	self.cnn_conv2 = nn.Conv2d(ninputChannels, nFilters[1], (filtsize[1], filtsize[1]), (1, 1))
	self.tanh2 = nn.Tanh()
	self.maxpool2 = nn.MaxPool2d((poolsize[1],poolsize[1]),(stepSize[1],stepSize[1])) 

	ninputChannels = nFilters[1]
	self.padding_3 = nn.ZeroPad2d(padDim)
	self.cnn_conv3 = nn.Conv2d(ninputChannels, nFilters[2], (filtsize[2], filtsize[2]), (1, 1))
	self.tanh3 = nn.Tanh()
	self.maxpool3 = nn.MaxPool2d((poolsize[2],poolsize[2]),(stepSize[2],stepSize[2])) 
	
	nFullyConnected = nFilters[2]*10*8


	self.non_local=NONLocalBlock2D(in_channels=32,inter_channels=64)

	self.cnn_drop = nn.Dropout2d(p=0.6)
	self.linear = nn.Linear(nFullyConnected,128)



	self.softmax = nn.Softmax(dim=0)

        self.score_fr = nn.Conv2d(4096, n_class, (1,1))
        self.upscore = nn.ConvTranspose2d(n_class, n_class, (1,64), stride=(1,32),
                                          bias=False)

	self.Sigmoid=nn.Sigmoid()

	self.classifierLayer = nn.Linear(128,150)

	self.logsoftmax=nn.LogSoftmax()

    def forward(self, x):
        cnn = self.padding_1(x)
	cnn = self.cnn_conv1(cnn)
	cnn = self.tanh1(cnn)
	cnn = self.maxpool1(cnn)

	cnn=self.padding_2(cnn)
	cnn = self.cnn_conv2(cnn)
	cnn = self.tanh2(cnn)
	cnn = self.maxpool2(cnn)

	cnn=self.padding_3(cnn)
	cnn = self.cnn_conv3(cnn)
	cnn = self.tanh3(cnn)
	cnn = self.maxpool3(cnn)
	nonlocal = cnn	

	attention_score = self.non_local(nonlocal)
	test = torch.rand((2,3))
	test1 = F.softmax(test,dim=0)
	ipdb.set_trace()
	
	
	nFullyConnected = 32*10*8  #nFilters[1]

	cnn = cnn.view(-1,nFullyConnected)
	cnn = self.cnn_drop(cnn)
	
	cnn = self.linear(cnn)
	
	cnn_transpose=cnn.transpose(0,1)

	attention_feature = torch.mm(cnn_transpose,attention_score)#128*16
	attention_feature = attention_feature.transpose(0,1)#16*128

	combine_feature = torch.add(cnn,attention_feature)#32*128

	combine_feature = torch.mean(combine_feature,0).unsqueeze(0)#1*128

	combine_feature = F.normalize(combine_feature, p=2, dim=1)

	classifier = self.classifierLayer(combine_feature)
	logsoftmax = self.logsoftmax(classifier)

        return combine_feature,logsoftmax


