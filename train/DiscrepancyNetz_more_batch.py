# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:48:01 2021

@author: UT111EM
"""

import torch
import torch.nn as nn
import os
from torchvision import models

'''
def torch_onehot(index, num_channels):

    size = index.shape
    oneHot_size = (size[0], num_channels, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, index.data.long().cuda(), 1.0)
    return input_label 

'''
def torch_onehot(index, num_channels, dtype=torch.uint8):
	"""
	Everything above num_channels will be all-0
	"""
	
	roi = index < num_channels
	index = index.byte().clone()
	index *= roi.byte()
	
	onehot = torch.zeros( 
		index.shape[:1] + (num_channels, ) + index.shape[1:], # add the channel dimension
		device = index.device,
		dtype = dtype,
	)
	# log.debug(f'one {onehot.shape} idx {index.shape} roi {roi.shape}')
	onehot.scatter_(1, index[:, None].long(), roi[:, None].type(dtype))
	return onehot


# VggFeatures
class VggFeatures_old(nn.Module):
    def __init__(self):
        super(VggFeatures, self).__init__()
                 
        self.VggFea1= nn.Sequential(
            
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(True), 
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(True)
        )    
        
        self.VggFea2= nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(True), 
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(True)
        )     
        
        self.VggFea3= nn.Sequential(
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(True), 
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(True)            
        )   
        
        self.VggFea4= nn.Sequential(
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(True), 
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(True)            
        )           
    def forward(self, x):
        
        outputs = []
        with torch.no_grad():
            x1 = self.VggFea1(x)
            x2 = self.VggFea2(x1)
            x3 = self.VggFea3(x2)
            x4 = self.VggFea4(x3)
        outputs.append(x1)
        outputs.append(x2)
        outputs.append(x3)
        outputs.append(x4)
        
        return outputs
        
        
# VggFeatures
class VggFeatures(nn.Module):
    def __init__(self,freeze=True):
    
        super(VggFeatures, self).__init__()
        
        
        vgg_mod = models.vgg16(pretrained=True)
        vgg_features = vgg_mod.features
        
        self.vgg_1 =  nn.Sequential(*vgg_features[0:4])
        self.vgg_2 =  nn.Sequential(*vgg_features[4:9])
        self.vgg_3 =  nn.Sequential(*vgg_features[9:16])
        self.vgg_4 =  nn.Sequential(*vgg_features[16:23])
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        
        
        
    def forward(self, labels):
        
        outputs = []
        x = labels
        x1 = self.vgg_1(x)
        x2 = self.vgg_2(x1)
        x3 = self.vgg_3(x2)
        x4 = self.vgg_4(x3)
        
        outputs.append(x1)
        outputs.append(x2)
        outputs.append(x3)
        outputs.append(x4)
        
        return outputs
    
'''
# Test VGG Features extractor    
vgg_extractor = VggFeatures()    
rand_tensor = torch.rand((4,3,320,320))
outs = vgg_extractor(rand_tensor)
print(outs[0].shape)
print(outs[1].shape)
print(outs[2].shape)
print(outs[3].shape)
'''

# SemFeatures
class SemFeatures(nn.Module):
    def __init__(self):
        super(SemFeatures, self).__init__()
                 
        self.SemFea1= nn.Sequential(
            nn.ReflectionPad2d((3, 3, 3, 3)),
            nn.Conv2d(in_channels=19,out_channels=32,kernel_size=(7,7),stride=(1,1)),
            nn.ReLU(True),        
        )    
        
        self.SemFea2= nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
            nn.ReLU(True), 
        )     
        
        self.SemFea3= nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
            nn.ReLU(True), 
        )    
        
        self.SemFea4= nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
            nn.ReLU(True), 
        )  
        
    def forward(self, labels):
        
        outputs = []
        #x = torch_onehot(labels, 19)
        #x = torch_onehot(labels, 19, dtype=torch.float32)
        x = labels
        x1 = self.SemFea1(x)
        x2 = self.SemFea2(x1)
        x3 = self.SemFea3(x2)
        x4 = self.SemFea4(x3)
        outputs.append(x1)
        outputs.append(x2)
        outputs.append(x3)
        outputs.append(x4)
        
        return outputs


'''
# Test SemFeatures
semFeatures = SemFeatures()   
torch.manual_seed(0)
rand_tensor = torch.rand((1,320,320))
outs = semFeatures(rand_tensor)

print(outs[0].shape)
print(outs[1].shape)
print(outs[2].shape)
print(outs[3].shape)
'''


# Correlation #
class CatMixCorr(nn.Module):
    def __init__(self):
        super().__init__()
                 
    def operation(self, a, b):          
        return torch.sum(a * b, dim=1, keepdim=True)  
     
    def forward(self, ImgfeatsList, GenImgfeatsList):
        
        CorrList = []
        x1 = self.operation(ImgfeatsList[3], GenImgfeatsList[3])
        CorrList.append(x1)
        x2 = self.operation(ImgfeatsList[2], GenImgfeatsList[2])
        CorrList.append(x2)
        x3 = self.operation(ImgfeatsList[1], GenImgfeatsList[1])
        CorrList.append(x3)   
        x4 = self.operation(ImgfeatsList[0], GenImgfeatsList[0])
        CorrList.append(x4)     
        
        return CorrList
	    
    
'''
# Test CatMixCorr
catMixCorr = CatMixCorr()   
CorrList = catMixCorr()   
'''

# Upsamling
class lxlConv(nn.Module):
    def __init__(self):
        super(lxlConv, self).__init__()
        
        self.conv_1x1_1 = nn.Conv2d(512 + 512 + 256,out_channels = 512,kernel_size = (1,1),stride = (1,1))
        self.conv_1x1_2 = nn.Conv2d(in_channels = 256 + 256 + 128,out_channels = 256,kernel_size = (1,1),stride = (1,1))
        self.conv_1x1_3 = nn.Conv2d(in_channels = 128 + 128 + 64,out_channels = 128,kernel_size = (1,1),stride = (1,1))
        self.conv_1x1_4 = nn.Conv2d(in_channels = 64 + 64 + 32 ,out_channels = 64,kernel_size = (1,1),stride = (1,1))
            
        
    def forward(self,ImgFeatsList, GenImgfeatsList,SegSemFeastList):
        lxlConvList = []
        x_input_1 = torch.cat([ImgFeatsList[3], GenImgfeatsList[3],SegSemFeastList[3]], 1)
        x1 = self.conv_1x1_1(x_input_1)
        lxlConvList.append(x1) 

        x_input_2 = torch.cat([ImgFeatsList[2], GenImgfeatsList[2],SegSemFeastList[2]], 1)
        x2 = self.conv_1x1_2(x_input_2)
        lxlConvList.append(x2) 

        x_input_3 = torch.cat([ImgFeatsList[1], GenImgfeatsList[1],SegSemFeastList[1]], 1)
        x3 = self.conv_1x1_3(x_input_3)
        lxlConvList.append(x3) 

        x_input_4 = torch.cat([ImgFeatsList[0], GenImgfeatsList[0],SegSemFeastList[0]], 1)
        x4 = self.conv_1x1_4(x_input_4)
        lxlConvList.append(x4)         
        
        return lxlConvList


'''
# Test lxlConv
lxlconv = lxlConv()   
print(lxlconv)

CorrList = lxlconv()   
'''


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
                 
        self.sigmoid = nn.Sigmoid()

    def Normalize(self,x):  
        xx = x.view(-1) 
        xx -= torch.min(xx)
        xx = xx
        xx /= torch.max(xx)  
        xx = xx.view(x.shape[0], x.shape[1], x.shape[2],x.shape[3]) 
        return xx


        
    def forward_old(self, CorrList,lxlConvList):
        
        
       
        lxlConvList_plus = []
        corrlistlist = CorrList.copy()
    

        #CorrList_norm_0 = self.Normalize(corrlistlist[0])
        muti = torch.mul(lxlConvList[0] , corrlistlist[0]) #CorrList_norm_0
        muti = self.sigmoid(muti)
        lxlConvList_plus.append(muti)
        
        #CorrList_norm_1 = self.Normalize(corrlistlist[1])
        muti= torch.mul(lxlConvList[1] , corrlistlist[1]) #CorrList_norm_1
        muti = self.sigmoid(muti)
        lxlConvList_plus.append(muti)
        
        #CorrList_norm_2 = self.Normalize(corrlistlist[2])
        muti = torch.mul(lxlConvList[2] , corrlistlist[2]) #CorrList_norm_2
        muti = self.sigmoid(muti)
        lxlConvList_plus.append(muti)
        
        #CorrList_norm_3 = self.Normalize(corrlistlist[3])
        muti= torch.mul(lxlConvList[3] , corrlistlist[3]) #CorrList_norm_3
        muti = self.sigmoid(muti)
        lxlConvList_plus.append(muti)
        

      
        return lxlConvList_plus
        
    def forward(self, CorrList,lxlConvList):
        
        
       
        lxlConvList_plus = []
        corrlist_0 = torch.clone(CorrList[0])
        corrlist_1 = torch.clone(CorrList[1])
        corrlist_2 = torch.clone(CorrList[2])
        corrlist_3 = torch.clone(CorrList[3])
        
        corrlist_0 = (corrlist_0 - torch.min(corrlist_0).item())/(torch.max(corrlist_0).item() - torch.min(corrlist_0).item())
        corrlist_1 = (corrlist_1 - torch.min(corrlist_1).item())/(torch.max(corrlist_1).item() - torch.min(corrlist_1).item())
        corrlist_2 = (corrlist_2 - torch.min(corrlist_2).item())/(torch.max(corrlist_2).item() - torch.min(corrlist_2).item())
        corrlist_3 = (corrlist_3 - torch.min(corrlist_3).item())/(torch.max(corrlist_3).item() - torch.min(corrlist_3).item())
    

       
        muti = torch.mul(lxlConvList[0], 1 - corrlist_0) #CorrList_norm_0
        #print(muti)
        #muti = self.sigmoid(muti)
        lxlConvList_plus.append(muti)
        
        
        muti= torch.mul(lxlConvList[1], 1 - corrlist_1) #CorrList_norm_1
        #print(muti)
        #muti = self.sigmoid(muti)
        lxlConvList_plus.append(muti)
        
        
        muti = torch.mul(lxlConvList[2], 1 - corrlist_2) #CorrList_norm_2
        #print(muti)
        #muti = self.sigmoid(muti)
        lxlConvList_plus.append(muti)
        
       
        muti= torch.mul(lxlConvList[3], 1 - corrlist_3) #CorrList_norm_3
        #print(muti)
        #muti = self.sigmoid(muti)
        lxlConvList_plus.append(muti)
        

      
        return lxlConvList_plus 
        
        


# Upsamling
class UpSampling(nn.Module):
    def __init__(self):
        super(UpSampling, self).__init__()
                 
        self.UpSamp1= nn.Sequential(
            nn.Conv2d(in_channels=513,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.SELU(True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.SELU(True),     
            nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=(2,2),stride=(2,2)),
        )    
        
        self.UpSamp2= nn.Sequential(
            nn.Conv2d(in_channels=513,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.SELU(True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.SELU(True),     
            nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=(2,2),stride=(2,2)),
        )      
        
        self.UpSamp3= nn.Sequential(
            nn.Conv2d(in_channels=385,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.SELU(True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.SELU(True),     
            nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=(2,2),stride=(2,2)),
        )      
        
        self.UpSamp4= nn.Sequential(
            nn.Conv2d(in_channels=193,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.SELU(True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.SELU(True),     
            
        )   
        
        self.final = nn.Conv2d(in_channels=64,out_channels=2,kernel_size=(1,1),stride=(1,1))

        
    def forward(self, CorrList, lxlConvList):
        
        x = torch.cat([CorrList[0], lxlConvList[0]], 1)
        x10 = self.UpSamp1(x)
        x1 = torch.cat([x10, CorrList[1], lxlConvList[1]], 1)
        x20 = self.UpSamp2(x1)
        x2 = torch.cat([x20, CorrList[2], lxlConvList[2]], 1)
        x30 = self.UpSamp3(x2)
        x3 = torch.cat([x30, CorrList[3], lxlConvList[3]], 1)
        x4 = self.UpSamp4(x3)
        outputs = self.final(x4)
        
        return outputs
        
'''
# Test Upsampling
Upsamp = UpSampling()   
print(Upsamp)
'''


class DiscrepancyNet(nn.Module):
    def __init__(self,att):
        super(DiscrepancyNet, self).__init__()
                 
        self.vgg_features = VggFeatures() 
        self.sem_features = SemFeatures()   
        self.cat_mix_corr = CatMixCorr() 
        self.lxl_conv = lxlConv() 
        self.upsampling = UpSampling()   
        self.attention = Attention()
        self.att = att
        
    def forward(self, image, gen_image, sem_label):
        
        image_feature = self.vgg_features(image)
        genImage_feature = self.vgg_features(gen_image)
        semLabel_feature = self.sem_features(sem_label)
        lxlConvList = self.lxl_conv(image_feature,genImage_feature,semLabel_feature)
        CorrList = self.cat_mix_corr(image_feature,genImage_feature) 
        if self.att:
            lxlConvList_plus = self.attention(CorrList,lxlConvList) 
        else:
            lxlConvList_plus = CorrList
        outputs = self.upsampling(CorrList,lxlConvList_plus)
        return outputs


# difference = DiscrepancyNet()
# torch.manual_seed(0)
# image = torch.rand((1,3,320,320))
# torch.manual_seed(1)
# gen = torch.rand((1,3,320,320))
# torch.manual_seed(2)
# sem_label = torch.rand((1,19,320,320))
# output = difference(image,gen,sem_label)

# print(image[0,0,0,1:10])
# print(gen[0,0,0,1:10])
# print(sem_label[0,0,1:10])
# print('\n')
# print(output[0,0,0,3:20])







