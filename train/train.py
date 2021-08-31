import argparse
import torch
#import torch.distributed as dist
#import yaml
import os
from tqdm import tqdm
import numpy as np
#import shutil
from PIL import Image
import time
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
#from torchvision.utils import make_grid
#from torchvision.transforms import ToPILImage, ToTensor

#from trainers.dissimilarity_trainer import DissimilarityTrainer
#from util import trainer_util

from contextlib import contextmanager
from image_dissimilarity.data.cityscapes_dataset import CityscapesDataset
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
#from image_dissimilarity.models.DiscrepancyNetz_more_batch import DiscrepancyNet
from DiscrepancyNetz_more_batch import DiscrepancyNet


#import torch.utils.data as data
from torch.utils.data import Dataset
from natsort import natsorted
from torchvision import transforms
import random

#from image_dissimilarity.data.augmentations import get_transform



#%% Config


path = '/home/xieyiru/xieyiru/DetectUnexcepteObjects/train'
#path = 'C:/Users/UT111EM/Desktop/Yiru/Signalverarbeitung/synbost-try'

dataloader = 'yolo'
config = {'experiment_name': 'test_0445', #Corrolation_muti_Features_test
         
          'save_folder': path + '/Result', 
          
          'training_strategy': {'niter': 60, 
                                'niter_decay': 0, 
                                'is_train': True, 
                                'class_weight': False, 
                                'class_weight_cityscapes': False, 
                                'image_visualization': False}, 
          
          'model': {'architecture': 'vgg16', 
                    'semantic': True, 
                    'pretrained': True, 
                    'correlation': True, 
                    'prior': False, 
                    'spade': 'decoder', 
                    'num_semantic_classes': 19}, 
          
          'diss_pretrained': {'load': 'initial',#'initial', 
                              #'load': 'continue',#'initial',  # continue
                              'save_folder': path + '/weight'
                             }, 
          
          'logger': {'results_dir': path + '/logs', 
                     'save_epoch_freq': 3}, 
         
          'loss': 'CrossEntropyLoss', 
         
          'optimizer': {'algorithm': 'Adam', 
                        
                        'parameters': {'lr': 0.0001, 
                                       'lr_policy': 'cosine',#'ReduceLROnPlateau', #
                                       'patience': 10, 
                                       'factor': 0.5, 
                                       'weight_decay': 0.0, 
                                       'beta1': 0.5, 
                                       'beta2': 0.999}}, 
          
          'train_dataloader': {'dataset_args': {'dataroot':  path + '/Dataset/Cityscapes',
                                                #'dataroot': '/home/xieyiru/xieyiru/Dataset/Dataset_Resize/CityScapes_Resize',# 
                                                'preprocess_mode': 'none', 
                                                'crop_size': 512, 
                                                'aspect_ratio': 2, 
                                                'flip': True, 
                                                'normalize': True, 
                                                'void': False, 
                                                'light_data': False, 
                                                'num_semantic_classes': 19, 
                                                'is_train': True, 
                                                'folders': 'all'}, 
                               
                               # 'dataloader_args': {'batch_size': 16, 
                               #                     'num_workers': 0, 
                               #                     'shuffle': True}}, 
          
                                'dataloader_args': {'batch_size': 32, 
                                                    'num_workers': 8, 
                                                    'shuffle': False}}, 
          
          

          
          
          'val_dataloader': {'dataset_args': {'dataroot':  path + '/Dataset/Cityscapes',
                                              #'dataroot':  '/home/xieyiru/xieyiru/Dataset/Dataset_Resize/CityScapes_Resize',
                                              #'dataroot': '/home/xieyiru/xieyiru/Dataset/Dataset_Resize/lost_and_found_Resize',
                                              'preprocess_mode': 'none', 
                                              'crop_size': 512, 
                                              'aspect_ratio': 2, 
                                              'flip': False, 
                                              'normalize': True, 
                                              'light_data': False, 
                                              'num_semantic_classes': 19, 
                                              'is_train': False, 
                                              'folders': 'train'}, 
                             
                             'dataloader_args': {'batch_size': 1, 
                                                 'num_workers': 8, 
                                                 'shuffle': False}}, 
          
          'test_dataloader1': {'dataset_args': {'dataroot':  path + '/Dataset/Lost_and_Found',
                                                #'dataroot': '/home/xieyiru/xieyiru/Dataset/Dataset_Resize/lost_and_found_Resize',
                                                'preprocess_mode': 'none', 
                                                'crop_size': 512, 
                                                'aspect_ratio': 2, 
                                                'flip': False, 
                                                'normalize': True, 
                                                'light_data': False, 
                                                'num_semantic_classes': 19, 
                                                'is_train': False, 
                                                'folders': 'train'}, 
                               
                               'dataloader_args': {'batch_size': 1, 
                                                   'num_workers': 0, 
                                                   'shuffle': False}}}
                                                   
                                                   

#%% Dataset_Cityscapes

# invalid frames are those where np.count_nonzero(labels_source) is 0 for Lost and Found Dataset
INVALID_LABELED_FRAMES = [17,  37,  55,  72,  91, 110, 129, 153, 174, 197, 218, 353, 490, 618, 686, 792, 793]

class CityscapesDataset(Dataset):
    
    def __init__(self, dataroot, preprocess_mode, crop_size=512, aspect_ratio= 0.5, flip=False, normalize=False,
                 prior = False, only_valid = False, roi = False, light_data= False, void = False, num_semantic_classes = 19, is_train = True,folders = 'train'):

        #dataset_list = []
        self.original_paths = []
        self.synthesis_paths = []
        self.semantic_paths = []
        self.synthesis_paths = []
        self.label_paths = []
        self.mae_features_paths = []
        self.entropy_paths = []
        self.logit_distance_paths = []
        
        if folders == 'train' or folders == 'val':
            dataset_list=[folders]
        elif folders == 'all':
            dataset_list = ['train','val']
        
        for i in range(len(dataset_list)):
            folder = dataset_list[i]
        
            self.original_paths += [os.path.join(dataroot,'original',folder, image)
                                   for image in os.listdir(os.path.join(dataroot,  'original',folder))]
            if light_data:
                self.semantic_paths += [os.path.join(dataroot, 'semantic_icnet',folder, image)
                                       for image in os.listdir(os.path.join(dataroot, 'semantic_icnet',folder))]
                self.synthesis_paths += [os.path.join(dataroot, 'synthesis_spade',folder, image)
                                        for image in os.listdir(os.path.join(dataroot, 'synthesis_spade',folder))]
            else:
                self.semantic_paths += [os.path.join(dataroot, 'semantic',folder, image)
                                       for image in os.listdir(os.path.join(dataroot, 'semantic',folder))]
                self.synthesis_paths += [os.path.join(dataroot, 'synthesis',folder, image)
                                        for image in os.listdir(os.path.join(dataroot, 'synthesis',folder))]
            if roi:
                self.label_paths += [os.path.join(dataroot, 'labels_with_ROI',folder, image)
                                    for image in os.listdir(os.path.join(dataroot, 'labels_with_ROI',folder))]
            elif void:
                self.label_paths += [os.path.join(dataroot, 'labels_with_void_no_ego',folder, image)
                                    for image in os.listdir(os.path.join(dataroot, 'labels_with_void_no_ego',folder))]
            else:
                self.label_paths += [os.path.join(dataroot, 'error_labels',folder, image)
                                    for image in os.listdir(os.path.join(dataroot, 'error_labels',folder))]
            if prior:
                if light_data:
                    self.mae_features_paths += [os.path.join(dataroot, 'mae_features_spade',folder, image)
                                               for image in os.listdir(os.path.join(dataroot, 'mae_features_spade',folder))]
                    self.entropy_paths += [os.path.join(dataroot, 'entropy_icnet',folder, image)
                                          for image in os.listdir(os.path.join(dataroot, 'entropy_icnet',folder))]
                    self.logit_distance_paths += [os.path.join(dataroot, 'logit_distance_icnet',folder, image)
                                                 for image in os.listdir(os.path.join(dataroot, 'logit_distance_icnet',folder))]
                else:
                    self.mae_features_paths += [os.path.join(dataroot, 'mae_features',folder, image)
                                               for image in os.listdir(os.path.join(dataroot, 'mae_features',folder))]
                    self.entropy_paths += [os.path.join(dataroot, 'entropy',folder, image)
                                          for image in os.listdir(os.path.join(dataroot, 'entropy',folder))]
                    self.logit_distance_paths += [os.path.join(dataroot, 'logit_distance',folder, image)
                                                 for image in os.listdir(os.path.join(dataroot, 'logit_distance',folder))]
            
            
        
        
        # We need to sort the images to ensure all the pairs match with each other
        self.original_paths = natsorted(self.original_paths)
        self.semantic_paths = natsorted(self.semantic_paths)
        self.synthesis_paths = natsorted(self.synthesis_paths)
        self.label_paths = natsorted(self.label_paths)
        

        
        
        
        if prior:
            self.mae_features_paths = natsorted(self.mae_features_paths)
            self.entropy_paths = natsorted(self.entropy_paths)
            self.logit_distance_paths = natsorted(self.logit_distance_paths)
        
        if only_valid: # Only for Lost and Found
            self.original_paths = np.delete(self.original_paths, INVALID_LABELED_FRAMES)
            self.semantic_paths = np.delete(self.semantic_paths, INVALID_LABELED_FRAMES)
            self.synthesis_paths = np.delete(self.label_paths, INVALID_LABELED_FRAMES)
            self.label_paths = np.delete(self.label_paths, INVALID_LABELED_FRAMES)
               
        assert len(self.original_paths) == len(self.semantic_paths) == len(self.synthesis_paths) \
               == len(self.label_paths), \
            "Number of images in the dataset does not match with each other"
        "The #images in %s and %s do not match. Is there something wrong?"
        
        
        
        self.original_path = self.read_img_from_path(self.original_paths,'Original Image','RGB')
        self.semantic_path = self.read_img_from_path(self.semantic_paths,'Semantic Image','L')
        self.synthesis_path = self.read_img_from_path(self.synthesis_paths,'Synthesis Image','RGB')
        self.label_path = self.read_img_from_path(self.label_paths,'Label','L')
        if prior:
            self.mae_features_path = self.read_img_from_path(self.mae_features_paths,'L')
            self.entropy_path = self.read_img_from_path(self.entropy_paths,'L')
            self.logit_distance_path = self.read_img_from_path(self.logit_distance_paths,'L')
                
            
        self.dataset_size = len(self.original_paths)
        self.preprocess_mode = preprocess_mode
        self.crop_size = crop_size
        self.aspect_ratio = aspect_ratio
        self.num_semantic_classes = num_semantic_classes
        self.is_train = is_train
        self.void = void
        self.flip = flip
        self.prior = prior
        self.normalize = normalize
        
    def read_img_from_path(self,path,name,Type):
        
        img_list = []
        length = len(path)
        pbar = tqdm(range(length),desc=name,position=0, leave=True)
        for i in pbar:
            img = Image.open(path[i]).convert(Type)
            img_list.append(img)
        return img_list
            
            
    def __getitem__(self, index):
        
        # get and open all images
        label = self.label_path[index]
        #label = Image.open(label_path)

        semantic = self.semantic_path[index]
        #semantic = Image.open(semantic_path)

        image = self.original_path[index]
        #image = Image.open(image_path).convert('RGB')

        syn_image = self.synthesis_path[index]
        #syn_image = Image.open(syn_image_path).convert('RGB')
        
        if self.prior:
            mae_image = self.mae_features_path[index]
            #mae_image = Image.open(mae_path)
    
            entropy_image = self.entropy_path[index]
            #entropy_image = Image.open(entropy_path)
    
            distance_image = self.logit_distance_path[index]
            #distance_image = Image.open(distance_path)

        # get input for transformations
        w = self.crop_size
        h = round(self.crop_size / self.aspect_ratio)
        image_size = (h, w)
        
        if self.flip:
            flip_ran = random.random() > 0.5
            label = _flip(label, flip_ran)
            semantic = _flip(semantic, flip_ran)
            image = _flip(image, flip_ran)
            syn_image = _flip(syn_image, flip_ran)
            if self.prior:
                mae_image = _flip(mae_image, flip_ran)
                entropy_image = _flip(entropy_image, flip_ran)
                distance_image = _flip(distance_image, flip_ran)

        # get augmentations
        base_transforms, augmentations = get_transform(image_size, self.preprocess_mode)

        # apply base transformations
        label_tensor = base_transforms(label)*255
        semantic_tensor = base_transforms(semantic)*255
        syn_image_tensor = base_transforms(syn_image)
        if self.prior:
            mae_tensor = base_transforms(mae_image)
            entropy_tensor = base_transforms(entropy_image)
            distance_tensor = base_transforms(distance_image)
        else:
            mae_tensor = []
            entropy_tensor = []
            distance_tensor = []

        if self.is_train and self.preprocess_mode != 'none':
            image_tensor = augmentations(image)
        else:
            image_tensor = base_transforms(image)
            
        if self.normalize:
            norm_transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) #imageNet normamlization
            syn_image_tensor = norm_transform(syn_image_tensor)
            image_tensor = norm_transform(image_tensor)
            
        # post processing for semantic labels
        if self.num_semantic_classes == 19:
            semantic_tensor[semantic_tensor == 255] = self.num_semantic_classes + 1  # 'ignore label is 20'
        semantic_tensor = one_hot_encoding(semantic_tensor, self.num_semantic_classes + 1)

        input_dict = {'label': label_tensor,
                      'original': image_tensor,
                      'semantic': semantic_tensor,
                      'synthesis': syn_image_tensor,
                      'label_path': self.label_paths[index],
                      'original_path': self.original_paths[index],
                      'semantic_path': self.semantic_paths[index],
                      'syn_image_path': self.synthesis_paths[index],
                      'entropy': entropy_tensor,
                      'mae': mae_tensor,
                      'distance': distance_tensor
                      }

        return input_dict
        
    def __len__(self):
        return self.dataset_size
    
def normalize():
    return

def _flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def one_hot_encoding(semantic, num_classes=20):
    one_hot = torch.zeros(num_classes, semantic.size(1), semantic.size(2))
    for class_id in range(num_classes):
        one_hot[class_id,:,:] = (semantic.squeeze(0)==class_id)
    one_hot = one_hot[:num_classes-1,:,:]
    return one_hot


#%% Transform

# from PIL import ImageFile
# from torchvision import transforms
# #from imgaug import augmenters as iaa
# #from imgaug import parameters as iap

# ImageFile.LOAD_TRUNCATED_IMAGES = True

# # defines all the different types of transformations
# class OnlyApplyBlurs:
#     def __init__(self):
#         self.aug = iaa.Sequential([iaa.Sometimes(0.25, iaa.OneOf([iaa.GaussianBlur(sigma=iap.Uniform(0, 3.0)),
#                                                                   iaa.MotionBlur(
#                                                                       k=iap.Choice([3, 7, 11, 15]), angle=0,
#                                                                       direction=1)]))
#                                    ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


# class OnlyApplyBlursMedium:
#     def __init__(self):
#         self.aug = iaa.Sequential([iaa.Sometimes(0.25, iaa.OneOf([iaa.GaussianBlur(sigma=iap.Uniform(0, 4.5)),
#                                                                   iaa.MotionBlur(
#                                                                       k=iap.Choice([11, 15, 21]), angle=0,
#                                                                       direction=1)]))
#                                    ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


# class OnlyApplyBlursStrong:
#     def __init__(self):
#         self.aug = iaa.Sequential([iaa.Sometimes(0.25, iaa.OneOf([iaa.GaussianBlur(sigma=iap.Uniform(0, 6.0)),
#                                                                   iaa.MotionBlur(
#                                                                       k=iap.Choice([15, 21, 27, 33]),
#                                                                       angle=0,
#                                                                       direction=1)]))
#                                    ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


# class OnlyApplyBlursAggressive:
#     def __init__(self):
#         self.aug = iaa.Sequential([iaa.Sometimes(0.50, iaa.OneOf([iaa.GaussianBlur(sigma=iap.Uniform(0, 8.0)),
#                                                                   iaa.MotionBlur(
#                                                                       k=iap.Normal(15, 50),
#                                                                       angle=iap.Normal(0, 360),
#                                                                       direction=iap.Normal(-1, 1))]))
#                                    ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


# class OnlyChangeContrast:
#     def __init__(self):
#         self.aug = iaa.Sequential(
#             [iaa.Sometimes(0.25, iaa.OneOf([iaa.contrast.LinearContrast(alpha=iap.Choice(np.arange(0, 3, 0.5).tolist())),
#                                             iaa.SigmoidContrast(gain=iap.Choice(np.arange(0, 3, 1).tolist()),
#                                                                 cutoff=iap.Choice(np.arange(0, 0.6, 0.10).tolist()))])),
#              ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


# class OnlyApplyDropout:
#     def __init__(self):
#         self.aug = iaa.Sequential([iaa.Sometimes(0.25, iaa.OneOf([iaa.Dropout(p=(0, 0.2)),
#                                                                   iaa.CoarseDropout(0.1, size_percent=0.25)]))
#                                    ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


# class OnlyApplyDropoutMedium:
#     def __init__(self):
#         self.aug = iaa.Sequential([iaa.Sometimes(0.25, iaa.OneOf([iaa.Dropout(p=(0, 0.35)),
#                                                                   iaa.CoarseDropout(0.15, size_percent=0.25)]))
#                                    ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


# class OnlyApplyDropoutStrong:
#     def __init__(self):
#         self.aug = iaa.Sequential([iaa.Sometimes(0.40, iaa.OneOf([iaa.Dropout(p=(0, 0.5)),
#                                                                   iaa.CoarseDropout(0.25, size_percent=0.25)]))
#                                    ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


# class OnlyApplyDropoutAggressive:
#     def __init__(self):
#         self.aug = iaa.Sequential([iaa.Sometimes(0.50, iaa.OneOf([iaa.Dropout(p=(0, 0.75)),
#                                                                   iaa.CoarseDropout(0.5, size_percent=0.5)]))
#                                    ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


# class OnlyApplyNoiseLight:
#     def __init__(self):
#         self.aug = iaa.Sequential(
#             [iaa.Sometimes(0.25, iaa.OneOf([iaa.AdditiveGaussianNoise((0, 0.1), (0, 0.1), per_channel=True),
#                                             iaa.AdditivePoissonNoise((0, 0.1), per_channel=True)]))
#              ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


# class OnlyApplyNoiseMedium:
#     def __init__(self):
#         self.aug = iaa.Sequential(
#             [iaa.Sometimes(0.25, iaa.OneOf([iaa.AdditiveGaussianNoise((0, 0.2), (0, 0.1), per_channel=True),
#                                             iaa.AdditivePoissonNoise((0, 0.2), per_channel=True)]))
#              ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


# class OnlyApplyNoiseStrong:
#     def __init__(self):
#         self.aug = iaa.Sequential(
#             [iaa.Sometimes(0.25, iaa.OneOf([iaa.AdditiveGaussianNoise((0, 0.2), (0, 0.2), per_channel=True),
#                                             iaa.AdditivePoissonNoise((0, 0.2), per_channel=True)]))
#              ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


# class OnlyApplyNoiseAggressive:
#     def __init__(self):
#         self.aug = iaa.Sequential(
#             [iaa.Sometimes(0.25, iaa.OneOf([iaa.AdditiveGaussianNoise((0, 100),
#                                                                       (0, 100),
#                                                                       per_channel=True),
#                                             iaa.AdditivePoissonNoise((0, 100),
#                                                                      per_channel=True)]))
#              ])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


# class OnlyApplyBrightnessAggressive:
#     def __init__(self):
#         self.aug = iaa.Sequential([iaa.Add(iap.Normal(-200, 200), per_channel=False)])

#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def get_transform(image_size, transform_name='blurs'):
    # uses ImageNet mean and standard deviation to normalize images
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    my_transforms = dict()
    #common_transforms = [transforms.Normalize(norm_mean, norm_std)]
    common_transforms = [transforms.Resize(size=image_size),transforms.ToTensor()]
    my_transforms['none'] = []
    my_transforms['base'] = transforms.Compose(common_transforms)
    my_transforms['normalization'] = transforms.Compose(common_transforms)
    # my_transforms['blurs'] = transforms.Compose([OnlyApplyBlurs(), lambda x: Image.fromarray(x)] + common_transforms)
    # my_transforms['contrast'] = transforms.Compose(
    #     [OnlyChangeContrast(), lambda x: Image.fromarray(x)] + common_transforms)
    # my_transforms['dropout'] = transforms.Compose(
    #     [OnlyApplyDropout(), lambda x: Image.fromarray(x)] + common_transforms)
    # my_transforms['color_jitter'] = transforms.Compose([transforms.ColorJitter(0.4, 0.4, 0.4)] + common_transforms)
    # my_transforms['color_jitter_dropout'] = transforms.Compose([OnlyApplyDropout(), lambda x: Image.fromarray(x)] +
    #                                                            [transforms.ColorJitter(0.4, 0.4, 0.4)] +
    #                                                            common_transforms)
    # my_transforms['geometry'] = transforms.Compose(
    #     [transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=(-25, +25)),
    #                              transforms.RandomRotation((-25, 25))], p=0.50)] + common_transforms)

    # my_transforms['all'] = transforms.Compose([OnlyApplyBlurs(),
    #                                            OnlyChangeContrast(),
    #                                            OnlyApplyDropout(), lambda x: Image.fromarray(x)] +
    #                                           [transforms.ColorJitter(0.4, 0.4, 0.4)] +
    #                                           common_transforms)

    # my_transforms['light_1'] = transforms.Compose([OnlyApplyDropout(),
    #                                                lambda x: Image.fromarray(x)] +
    #                                               [transforms.ColorJitter(0.3, 0.3, 0.3)] +
    #                                               common_transforms)

    # my_transforms['light_2'] = transforms.Compose([OnlyApplyBlurs(),
    #                                                lambda x: Image.fromarray(x)] +
    #                                               [transforms.ColorJitter(0.3, 0.3, 0.3)] +
    #                                               common_transforms)

    # my_transforms['light_3'] = transforms.Compose([OnlyApplyBlurs(),
    #                                                OnlyApplyNoiseMedium(),
    #                                                lambda x: Image.fromarray(x)] +
    #                                               [transforms.ColorJitter(0.3, 0.3, 0.3)] +
    #                                               common_transforms)

    # my_transforms['medium_1'] = transforms.Compose([OnlyApplyDropoutMedium(),
    #                                                 lambda x: Image.fromarray(x)] +
    #                                                [transforms.ColorJitter(0.4, 0.4, 0.4)] +
    #                                                common_transforms)

    # my_transforms['medium_2'] = transforms.Compose([OnlyApplyBlursMedium(),
    #                                                 lambda x: Image.fromarray(x)] +
    #                                                [transforms.ColorJitter(0.4, 0.4, 0.4)] +
    #                                                common_transforms)

    # my_transforms['medium_3'] = transforms.Compose([OnlyApplyBlursMedium(),
    #                                                 OnlyApplyNoiseMedium(),
    #                                                 lambda x: Image.fromarray(x)] +
    #                                                [transforms.ColorJitter(0.4, 0.4, 0.4)] +
    #                                                common_transforms)

    # my_transforms['medium_4'] = transforms.Compose([OnlyApplyBlursMedium(),
    #                                                 OnlyApplyDropoutMedium(),
    #                                                 OnlyApplyNoiseMedium(),
    #                                                 lambda x: Image.fromarray(x)] +
    #                                                [transforms.ColorJitter(0.4, 0.4, 0.4)] +
    #                                                common_transforms)

    # my_transforms['strong_1'] = transforms.Compose([OnlyApplyBlursStrong(),
    #                                                 OnlyApplyDropoutStrong(),
    #                                                 lambda x: Image.fromarray(x)] +
    #                                                [transforms.ColorJitter(0.5, 0.5, 0.5)] +
    #                                                common_transforms)

    # my_transforms['strong_2'] = transforms.Compose([OnlyApplyBlursStrong(),
    #                                                 OnlyApplyDropoutStrong(),
    #                                                 lambda x: Image.fromarray(x)] +
    #                                                [transforms.ColorJitter(0.5, 0.5, 0.5)] +
    #                                                common_transforms)

    # my_transforms['strong_3'] = transforms.Compose([OnlyApplyBlursStrong(),
    #                                                 OnlyApplyNoiseMedium(),
    #                                                 lambda x: Image.fromarray(x)] +
    #                                                [transforms.ColorJitter(0.5, 0.5, 0.5)] +
    #                                                common_transforms)

    # my_transforms['strong_4'] = transforms.Compose([OnlyApplyBlursStrong(),
    #                                                 OnlyApplyDropoutStrong(),
    #                                                 OnlyApplyNoiseMedium(),
    #                                                 lambda x: Image.fromarray(x)] +
    #                                                [transforms.ColorJitter(0.5, 0.5, 0.5)] +
    #                                                common_transforms)

    return my_transforms['base'], my_transforms[transform_name]





#%% select_device
def select_device(device='1', batch_size=None, logger = None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
    
    return torch.device('cuda:0' if cuda else 'cpu')



#%% Yolo Dataloader
class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)    
    
@contextmanager    
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()
        

def get_dataloader(dataset_args, dataloader_args):
    
    with torch_distributed_zero_first(1):
        dataset = CityscapesDataset(**dataset_args)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) 
    dataloader = InfiniteDataLoader(dataset,
                                batch_size=dataloader_args['batch_size'],
                                num_workers=8,
                                sampler=sampler,
                                pin_memory=True)  # torch.utils.data.DataLoader()    

    return dataloader   

def get_dataloader_old(dataset_args, dataloader_args):
    dataset = CityscapesDataset(**dataset_args)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    
    return dataloader    



#%% Select the device
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend = 'nccl',init_method='tcp://10.130.121.13:23460',world_size=1,rank=0)
device = select_device(device = '0', batch_size=4)


# get experiment information
exp_name = config['experiment_name']
save_fdr = config['save_folder']
logs_fdr = config['logger']['results_dir']

print('Starting experiment named: %s'%exp_name)

if not os.path.isdir(save_fdr):
    os.mkdir(save_fdr)

if not os.path.isdir(logs_fdr):
    os.mkdir(logs_fdr)
    
    
#%% Initiale SummaryWriter
Sum_writer = SummaryWriter(os.path.join(logs_fdr, exp_name), flush_secs=30)
Test_writer = SummaryWriter(os.path.join(logs_fdr, exp_name), flush_secs=30)
# Activate GPUs
#config['gpu_ids'] = opts.gpu_ids
#gpu_info = trainer_util.activate_gpus(config)



#%% Dataloader Setting
# Get data loaders
cfg_train_loader = config['train_dataloader']
cfg_val_loader = config['val_dataloader']
cfg_test_loader1 = config['test_dataloader1']
# cfg_test_loader2 = config['test_dataloader2']
# cfg_test_loader3 = config['test_dataloader3']
# cfg_test_loader4 = config['test_dataloader4']


# checks if we are using prior images
prior = config['model']['prior']
# adds logic to dataloaders (avoid repetition in config file)
cfg_train_loader['dataset_args']['prior'] = prior
cfg_val_loader['dataset_args']['prior'] = prior
cfg_test_loader1['dataset_args']['prior'] = prior
# cfg_test_loader2['dataset_args']['prior'] = prior
# cfg_test_loader3['dataset_args']['prior'] = prior
# cfg_test_loader4['dataset_args']['prior'] = prior
    

if dataloader == 'yolo':
    train_loader = get_dataloader(cfg_train_loader['dataset_args'], cfg_train_loader['dataloader_args'])
    val_loader = get_dataloader(cfg_val_loader['dataset_args'], cfg_val_loader['dataloader_args'])
    test_loader1 = get_dataloader(cfg_test_loader1['dataset_args'], cfg_test_loader1['dataloader_args'])
else:
    train_loader = get_dataloader_old(cfg_train_loader['dataset_args'], cfg_train_loader['dataloader_args'])
    val_loader = get_dataloader_old(cfg_val_loader['dataset_args'], cfg_val_loader['dataloader_args'])
    test_loader1 = get_dataloader_old(cfg_test_loader1['dataset_args'], cfg_test_loader1['dataloader_args'])

#train_loader = trainer_util.get_dataloader(cfg_train_loader['dataset_args'], cfg_train_loader['dataloader_args'])
#val_loader = trainer_util.get_dataloader(cfg_val_loader['dataset_args'], cfg_val_loader['dataloader_args'])
#test_loader1 = trainer_util.get_dataloader(cfg_test_loader1['dataset_args'], cfg_test_loader1['dataloader_args'])
# test_loader2 = trainer_util.get_dataloader(cfg_test_loader2['dataset_args'], cfg_test_loader2['dataloader_args'])
# test_loader3 = trainer_util.get_dataloader(cfg_test_loader3['dataset_args'], cfg_test_loader3['dataloader_args'])
# test_loader4 = trainer_util.get_dataloader(cfg_test_loader4['dataset_args'], cfg_test_loader4['dataloader_args'])


# Getting parameters for test
dataset = cfg_val_loader['dataset_args']
h = int((dataset['crop_size']/dataset['aspect_ratio']))
w = int(dataset['crop_size'])


#%% Model aufbauen

class DissimilarityTrainer(nn.Module):
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, config, device,seed=0):
        
        super(DissimilarityTrainer, self).__init__()
        self.set_seed(seed)
        
        cudnn.enabled = True
        self.config = config 
        self.device = device
        
        self.diss_model = DiscrepancyNet(False).to(self.device)
        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            self.diss_model = torch.nn.DataParallel(self.diss_model,device_ids = [0])
        else:
            raise NotImplementedError()


        #print('Printing Model Parameters')
        #print(self.diss_model.parameters)
        
        lr_config = config['optimizer']
        lr_options = lr_config['parameters']
        
        if lr_config['algorithm'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.diss_model.parameters(), lr=lr_options['lr'],
                                             weight_decay=lr_options['weight_decay'],)
        elif lr_config['algorithm'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.diss_model.parameters(),
                                              lr=lr_options['lr'],
                                              weight_decay=lr_options['weight_decay'],
                                              betas=(lr_options['beta1'], lr_options['beta2']))
        else:
            raise NotImplementedError
            
        if lr_options['lr_policy'] == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=lr_options['patience'], factor=lr_options['factor'])
        
        elif lr_options['lr_policy'] == 'cosine':
            lf = lambda x: ((1 + math.cos(x * math.pi / config['training_strategy']['niter'])) / 2) * (1 - 0.2) + 0.2  # cosine
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf) 

        else:
            raise NotImplementedError
        
        self.old_lr = lr_options['lr']
        self.lr_policy = lr_options['lr_policy'] 
        

        
        
                # get pre-trained model
        pretrain_config = config['diss_pretrained']
        self.save_ckpt_fdr = pretrain_config['save_folder']
        self.ckpt_name = config['experiment_name']
        
        if pretrain_config['load'] == 'initial':
            
            
            # path = '/home/xieyiru/xieyiru/synbost-try/weight/Test_Original'
            # Savepath = path + '/Test_Original.pth'
            
            # if not os.path.exists(Savepath):
            #     torch.save(self.diss_model.state_dict(),path + '/temp.pth')
            #     dict_self = torch.load(path + '/temp.pth')
            #     dict_pr = torch.load(path + '/Test_Original_.pth')
            #     dict_pre = dict_pr#['weights']
            #     dict_diff = dict_self.copy()
            #     LayerName_gan = list(dict_diff.keys())
            #     for i, key in enumerate(dict_pre.keys()):
            #         assert torch.numel(dict_diff[LayerName_gan[i]]) == torch.numel(dict_pre[key])
            #         dict_diff[list(dict_diff.keys())[i]]=dict_pre[key]
            #     torch.save(dict_diff,Savepath)  
            
            #print('load the pretrained weight...')    
            #self.diss_model.load_state_dict(torch.load(Savepath)) 
                        
     
            print('Loading pretrained weights')
            model_path = os.path.join(self.save_ckpt_fdr, 'baseline_cosine_vgg_pretrained_1008', 'baseline_cosine_vgg_pretrained_1008.pth')
            #model_path = os.path.join(self.save_ckpt_fdr, 'Test_Original', 'Test_Original.pth')            
            model_weights = torch.load(model_path)
            #self.diss_model.load_state_dict(model_weights)
            self.diss_model.load_state_dict(model_weights['model'])
            self.epoch = 1
            
         
        elif pretrain_config['load'] == 'continue': 

            path = os.path.join(self.save_ckpt_fdr,self.ckpt_name,self.ckpt_name + '.pth')
            
            self.Info = torch.load(path)
            model_weights = self.Info['model']
            self.diss_model.load_state_dict(model_weights)
            self.epoch = self.Info['epoch']
            optimizer_weights = self.Info['optimizer']
            self.optimizer.load_state_dict(optimizer_weights)
            
            
            
        #     epoch = pretrain_config['which_epoch']
        #     self.save_ckpt_fdr = pretrain_config['save_folder']
        #     self.ckpt_name = pretrain_config['experiment_name']

        #     print('Loading pretrained weights from %s (epoch: %s)' % (self.ckpt_name, epoch))
        #     model_path = os.path.join(self.save_ckpt_fdr, self.ckpt_name, '%s_net_%s.pth' % (epoch, self.ckpt_name))
        #     
        #     self.diss_model.load_state_dict(model_weights, strict=False)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=255).to(self.device)
        
    def set_seed(self,seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    def get_model(self):
        model = self.diss_model
        return model
    
    def run_model_one_step(self, inputs, label):
        self.optimizer.zero_grad()
        predictions = self.diss_model(inputs)
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).to(self.device))
        #torch.autograd.set_detect_anomaly(True)
        model_loss.backward()
        self.optimizer.step()
        self.model_losses = model_loss
        self.generated = predictions
        return model_loss, predictions
        
    def run_validation(self, inputs, label):
        predictions = self.diss_model(inputs)
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).to(self.device))
        return model_loss, predictions

    def run_model_one_step_prior(self, original, synthesis, semantic, label, entropy, mae, distance):
        self.optimizer.zero_grad()
        predictions = self.diss_model(original, synthesis, semantic, entropy, mae, distance)
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).to(self.device))
        model_loss.backward()
        self.optimizer.step()
        self.model_losses = model_loss
        self.generated = predictions
        return model_loss, predictions

    def run_validation_prior(self, original, synthesis, semantic, label, entropy, mae, distance):
        predictions = self.diss_model(original, synthesis, semantic, entropy, mae, distance)
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).to(self.device))
        return model_loss, predictions

    def get_latest_losses(self):
        return {**self.model_loss}

    def get_latest_generated(self):
        return self.generated
    
    def record_epoch(self,epoch):
        self.epoch = epoch     
        
    def get_epoch(self):
        return self.epoch
     

    def save(self, save_dir, epoch, name,result):
        if not os.path.isdir(os.path.join(save_dir, name)):
            os.mkdir(os.path.join(save_dir, name))
        
        #save_filename = '%s_net_%s.pth' % (epoch, name)
        #save_path = os.path.join(save_dir, name, save_filename)
        #torch.save(self.diss_model.state_dict(), save_path)  # net.cpu() -> net

        Save_Info = {'model': self.diss_model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'result': result,
                      'epoch':self.epoch
                      }
                    
    
        #save_path_c = os.path.join(save_dir, name, save_filename_c)
       
        if not os.path.exists(os.path.join(self.save_ckpt_fdr,self.ckpt_name)):
            os.makedirs(os.path.join(self.save_ckpt_fdr,self.ckpt_name))
            
        save_path_c = os.path.join(self.save_ckpt_fdr,self.ckpt_name,self.ckpt_name + '.pth')
        torch.save(Save_Info, save_path_c)



#%% Load the Model
# create trainer for our model
print('Loading Model')
trainer = DissimilarityTrainer(config,device)

# create tool for counting iterations
batch_size = config['train_dataloader']['dataloader_args']['batch_size']
#iter_counter = IterationCounter(config, len(train_loader), batch_size)

# Softmax layer for testing
softmax = torch.nn.Softmax(dim=1)




#%% Evaluation_Matrix


def get_metrics(flat_labels, flat_pred, num_points=50):
    # From fishycapes code
    pos = flat_labels == 1
    valid = flat_labels <= 1  # filter out void
    gt = pos[valid]  # remove the outline elements
    del pos
    uncertainty = flat_pred[valid].reshape(-1).astype(np.float32, copy=False)
    del valid

    # Sort the classifier scores (uncertainties)
    sorted_indices = np.argsort(uncertainty, kind='mergesort')[::-1]
    uncertainty, gt = uncertainty[sorted_indices], gt[sorted_indices]
    del sorted_indices
    
    # Remove duplicates along the curve
    distinct_value_indices = np.where(np.diff(uncertainty))[0]
    threshold_idxs = np.r_[distinct_value_indices, gt.size - 1]
    del distinct_value_indices, uncertainty
    
    # Accumulate TPs and FPs
    tps = np.cumsum(gt, dtype=np.uint64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    del threshold_idxs
    
    if tps[-1] == 0:
       tps[-1] = 0.00001 
    if fps[-1] == 0:
       fps[-1] = 0.00001  
    
    # Compute Precision and Recall
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained and reverse the outputs so recall is decreasing
    sl = slice(tps.searchsorted(tps[-1]), None, -1)
    precision = np.r_[precision[sl], 1]
    recall = np.r_[recall[sl], 0]
    average_precision = -np.sum(np.diff(recall) * precision[:-1])
    
    # select num_points values for a plotted curve
    interval = 1.0 / num_points
    curve_precision = [precision[-1]]
    curve_recall = [recall[-1]]
    idx = recall.size - 1
    for p in range(1, num_points):
        while recall[idx] < p * interval:
            idx -= 1
        curve_precision.append(precision[idx])
        curve_recall.append(recall[idx])
    curve_precision.append(precision[0])
    curve_recall.append(recall[0])
    del precision, recall
    
    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0., tps]
        fps = np.r_[0., fps]
    
    
   
    # Compute TPR and FPR
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    del tps
    del fps
    
    # Compute AUROC
    auroc = np.trapz(tpr, fpr)   #Integrate along the given axis using the composite trapezoidal rule.
    
    # Compute FPR@95%TPR
    fpr_tpr95 = fpr[np.searchsorted(tpr, 0.95)]
    results = {
        'auroc': auroc,
        'AP': average_precision,
        'FPR@95%TPR': fpr_tpr95,
        'recall': np.array(curve_recall),
        'precision': np.array(curve_precision),
        'fpr': fpr,
        'tpr': tpr
        }

    return results

#%% train
def train(trainer,train_loader,Sum_writer,val_loader,softmax,w,h,cfg_val_loader,save_fdr,exp_name):
    print('Starting Training...')
    
    best_val_loss = float('inf')
    best_map_metric = 0
    Iter = 0
    
    first_epoch = trainer.get_epoch()
    end_epoch = first_epoch + config['training_strategy']['niter']

    for epoch in range(first_epoch,end_epoch):
    
    #print('Starting Epoch %i for experiment %s'% (epoch, exp_name))
    #iter_counter.record_epoch_start(epoch)
        train_loss = 0
        
        t_train = tqdm(train_loader,position=0, leave=True)
        trainer.record_epoch(epoch)
        for i, data_i in enumerate(t_train):  #, start=iter_counter.epoch_iter
            #iter_counter.record_one_iteration()
            original = data_i['original'].to(device)
            semantic = data_i['semantic'].to(device)
            synthesis = data_i['synthesis'].to(device)
            label = data_i['label'].to(device)
            inputs = torch.cat([original,synthesis,semantic],dim = 1)
            # Training
            model_loss, _ = trainer.run_model_one_step(inputs, label)
                
            train_loss += model_loss
            Sum_writer.add_scalar('Train/Train_Loss_iter', model_loss, Iter)
            Iter+=1
            
            t_train.set_description('Train: Epoch %i  Loss: %.2f'% (epoch,model_loss))
            t_train.refresh() # to show immediately the update
                
            
        avg_train_loss = train_loss / len(train_loader)
        Sum_writer.add_scalar('Train/Train_Loss', avg_train_loss, epoch)
        
        
        
        results = Validation(val_loader,trainer,epoch,softmax,w,h,Sum_writer,cfg_val_loader)
        
            
        trainer.save(save_fdr, 'latest', exp_name, results)
        
        lr = trainer.get_learning_rate()
        #lr = config['optimizer']['parameters']['lr']
        Sum_writer.add_scalar('Lr/Learning_Rate', lr, epoch)
        
        trainer.update_learning_rate_schedule(results['avg_val_loss'])
                #print('Current learning rate is set for %f' %lr)
        
        
#%% Validation        
        
def Validation(val_loader,trainer,epoch,softmax,w,h,Sum_writer,cfg_val_loader):
    
    
    #if epoch >10:
    if True:    
        
        flat_pred = np.zeros(w * h * len(val_loader))#.to(device)
        flat_labels = np.zeros(w * h * len(val_loader))#.to(device)
        
    with torch.no_grad():
        results = {}    
        val_loss = 0
        t_val = tqdm(val_loader,position=0, leave=True)
        for i, data_i in enumerate(t_val):
            s1 = time.time()
            original = data_i['original'].to(device)
            semantic = data_i['semantic'].to(device)
            synthesis = data_i['synthesis'].to(device)
            label = data_i['label'].to(device)
            inputs = torch.cat([original,synthesis,semantic],dim = 1)
            loss, outputs = trainer.run_validation(inputs, label)

            val_loss += loss
            
            
            
            
            t_val.set_description('Validation: Epoch %i  Loss: %.2f'% (epoch,loss))
            t_val.refresh() # to show immediately the update
            
            outputs = softmax(outputs)
            (softmax_pred, predictions) = torch.max(outputs, dim=1)
            avg_val_loss = val_loss / len(val_loader)
            #if epoch >10:
            if True:      
                flat_pred[i * w * h:i * w * h + w * h] = torch.flatten(outputs[:, 1, :, :]).detach().cpu().numpy()
                flat_labels[i * w * h:i * w * h + w * h] = torch.flatten(label).detach().cpu().numpy()
                #print('hi')
            

       
        if True: 
            results = get_metrics(flat_labels, flat_pred)
            Sum_writer.add_scalar('Validation/AUC_ROC_%s' % os.path.basename(cfg_val_loader['dataset_args']['dataroot']), results['auroc'], epoch)
            Sum_writer.add_scalar('Validation/mAP_%s' % os.path.basename(cfg_val_loader['dataset_args']['dataroot']), results['AP'], epoch)
            Sum_writer.add_scalar('Validation/FPR@95TPR_%s' % os.path.basename(cfg_val_loader['dataset_args']['dataroot']), results['FPR@95%TPR'], epoch)
            avg_val_loss = val_loss / len(val_loader)  
            results['avg_val_loss'] = avg_val_loss
            Sum_writer.add_scalar('Validation/Val_loss_%s' % os.path.basename(cfg_val_loader['dataset_args']['dataroot']), avg_val_loss, epoch)
       
        else:
            avg_val_loss = val_loss / len(val_loader)  
            results['avg_val_loss'] = avg_val_loss
            Sum_writer.add_scalar('Validation/Val_loss_%s' % os.path.basename(cfg_val_loader['dataset_args']['dataroot']), avg_val_loss, epoch)

        return results  
    

#%% Test 
    
def Test(val_loader,trainer,epoch,softmax,w,h,Sum_writer,cfg_val_loader):
    
    
    #if epoch >10:
    if True:    
        
        flat_pred = np.zeros(w * h * len(val_loader))#.to(device)
        flat_labels = np.zeros(w * h * len(val_loader))#.to(device)
        
    with torch.no_grad():
        results = {}    
        val_loss = 0
        t_val = tqdm(val_loader,position=0, leave=True)
        for i, data_i in enumerate(t_val):
            s1 = time.time()
            original = data_i['original'].to(device)
            semantic = data_i['semantic'].to(device)
            synthesis = data_i['synthesis'].to(device)
            label = data_i['label'].to(device)
            inputs = torch.cat([original,synthesis,semantic],dim = 1)

            loss, outputs = trainer.run_validation(inputs, label)

            val_loss += loss
            
            
            
            
            t_val.set_description('Validation: Epoch %i  Loss: %.2f'% (epoch,loss))
            t_val.refresh() # to show immediately the update
            
            outputs = softmax(outputs)
            (softmax_pred, predictions) = torch.max(outputs, dim=1)
            avg_val_loss = val_loss / len(val_loader)
            #semanti = semantic[:,0,:,:]
            outputs = outputs[:,1,:,:]
            
            #output = (outputs[0,:,:] + 1) * 128
            #output = torch.clamp(output, 0, 255)
            # labels = label[0,0,:,:].type(torch.uint8).cpu().numpy()
            # labels = np.asarray(labels)
            # plt.imshow(labels)
            
           
            
            output = (outputs[0,:,:] + 1) * 128
            a = outputs[0,:,:]
            output = torch.clamp(output, 0, 255)
            output = output.type(torch.uint8).cpu().numpy()
            #output = outputs[0,:,:].type(torch.uint8).cpu().numpy()
            output = np.asarray(output)
            plt.imshow(output) 
           
            #output[output<0.6] = 0
            #output[output>=0.6] = 1
            #output[semanti == 0] = 0
            #if epoch >10
            if True:     
                #flat_pred[i * w * h:i * w * h + w * h] = torch.flatten(output).detach().cpu().numpy()
                flat_pred[i * w * h:i * w * h + w * h] = torch.flatten(outputs).detach().cpu().numpy()
                flat_labels[i * w * h:i * w * h + w * h] = torch.flatten(label).detach().cpu().numpy()
                #print('hi')
            

       
        if True: 
            results = get_metrics(flat_labels, flat_pred)
            #Test_writer.add_graph(trainer.get_model, inputs)
            #Sum_writer.add_scalar('Validation/AUC_ROC_%s' % os.path.basename(cfg_val_loader['dataset_args']['dataroot']), results['auroc'], epoch)
            #Sum_writer.add_scalar('Validation/mAP_%s' % os.path.basename(cfg_val_loader['dataset_args']['dataroot']), results['AP'], epoch)
            #Sum_writer.add_scalar('Validation/FPR@95TPR_%s' % os.path.basename(cfg_val_loader['dataset_args']['dataroot']), results['FPR@95%TPR'], epoch)
            avg_val_loss = val_loss / len(val_loader)  
            results['avg_val_loss'] = avg_val_loss
            #Sum_writer.add_scalar('Validation/Val_loss_%s' % os.path.basename(cfg_val_loader['dataset_args']['dataroot']), avg_val_loss, epoch)
       
        else:
            avg_val_loss = val_loss / len(val_loader)  
            results['avg_val_loss'] = avg_val_loss
            Sum_writer.add_scalar('Validation/Val_loss_%s' % os.path.basename(cfg_val_loader['dataset_args']['dataroot']), avg_val_loss, epoch)

        return results  
 



#%% main function
       
#train(trainer,train_loader,Sum_writer,val_loader,softmax,w,h,cfg_val_loader,save_fdr,exp_name)  


# first_epoch = trainer.get_epoch()
# end_epoch = first_epoch + config['training_strategy']['niter']

# for epoch in range(first_epoch,end_epoch):
#     Validation(val_loader,trainer,epoch,softmax,w,h,Sum_writer,cfg_val_loader)





#################################
#Testing
epoch = 1
results = Test(test_loader1,trainer,epoch,softmax,w,h,Sum_writer,cfg_val_loader)

print('Test: Train/Training Loss: %f' % (results['avg_val_loss'])) 
print('Test: AU_ROC: %f' % results['auroc'])
print('Test: mAP: %f' % results['AP'])
print('Test: FPR@95TPR: %f' % results['FPR@95%TPR'])

#####################################









