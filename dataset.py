import os
from pathlib import Path
import numpy as np

from PIL import Image
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm as tqdm
import cv2

import torch
import random

import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

import utils

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from skimage import morphology

def create_data(train_data):
    print(train_data)
    tumor_set, stroma_set, normal_set = set(), set(), set()
    for path in Path(train_data).glob('*.png'):
        if utils.is_tumor(path): tumor_set.add(str(path))
        if utils.is_stroma(path): stroma_set.add(str(path))
        if utils.is_normal(path): normal_set.add(str(path))

    tumor_images = list(tumor_set - stroma_set - normal_set)
    stroma_images = list(stroma_set - tumor_set - normal_set)
    normal_images = list(normal_set - tumor_set - stroma_set)

    return tumor_images, stroma_images, normal_images

class LabeledDataset(BaseDataset):
    def __init__(self, image_dir, mask_dir, transforms=None, num_classes=3):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.name_list = sorted([i for i in os.listdir(self.image_dir) if i.endswith('.png')])
        
        if transforms is None:
            self.transforms = albu.Compose([
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(transpose_mask=True),
            ])
        else:
            self.transforms = transforms
        
        self.num_classes = num_classes

    def __getitem__(self, i):
        name = self.name_list[i]
        # print(name, i)
        image = np.array(Image.open(os.path.join(self.image_dir, name)))
        mask = np.array(Image.open(os.path.join(self.mask_dir, name)), dtype=np.uint8)
        
        sample = self.transforms(image=image, mask=mask)

        label = [0] * self.num_classes
        category = np.unique(mask)
        for i in range(self.num_classes):
            if i in category:
                label[i] = 1

        return {'image': sample['image'], 'mask': sample['mask'], 'label': torch.Tensor(label), 'name': name, 'h': image.shape[0], 'w': image.shape[1]}
    
    def __len__(self):
        return len(self.name_list)


class UnlabeledDataset(BaseDataset):
    def __init__(self, image_dir, transforms=None, num_classes=3):
        self.image_dir = image_dir
        self.name_list = sorted([i for i in os.listdir(self.image_dir) if i.endswith('.png')])
        
        if transforms is None:
            self.transforms = transforms.Compose([
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms
        
        self.num_classes = num_classes

    def __getitem__(self, i):
        name = self.name_list[i]
        label = self.get_label(name)
        image = np.array(Image.open(os.path.join(self.image_dir, name)))
        image = self.transforms(image=image)['image']

        return {'image': image, 'label': torch.Tensor(label), 'name': name, 'h': image.shape[0], 'w': image.shape[1]}
    
    def get_label(self, name):
        label_str = '[' + name.split('[')[-1].split(']')[0] + ']'
        if ' ' not in label_str:
            label_str = '[' + ' '.join([i for i in label_str[1:-1]]) + ']'
        if ',' not in label_str:
            label_str = label_str.replace(' ', ',')
        label = eval(label_str)
        return label
    
    def __len__(self):
        return len(self.name_list)
        
class TestDataset(BaseDataset):
    def __init__(self, args):
        self.args = args
        self.test_data = Path(args.test_data)
        self.test_image = sorted(list((self.test_data / 'img').glob('*.png')))
        self.transforms = albu.Compose([
            albu.PadIfNeeded(self.args.patch_size, self.args.patch_size, border_mode=2, position=albu.PadIfNeeded.PositionType.TOP_LEFT),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(transpose_mask=True),
        ])


    def __getitem__(self, i):
        name = self.test_image[i].name
        image = self.test_image[i]
        image = np.array(Image.open(image))
        mask = np.array(Image.open(self.test_data / 'mask' / name))
        original_h, original_w = image.shape[:2]

        sample = self.transforms(image=image, mask=mask)
        image, mask = sample['image'], sample['mask'].long()

        return image, mask, name, original_h, original_w

    def __len__(self):
        return len(self.test_image)

class TrainDataset(BaseDataset):
    def __init__(self, args):
        self.args = args

        train_data = Path(args.train_data)
        self.train_image = sorted(list(train_data.glob('*.png')))

        self.transforms = albu.Compose([
            albu.Resize(args.patch_size, args.patch_size),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(transpose_mask=True),
        ])

    def __getitem__(self, i):
        name = self.train_image[i].name
        image = np.array(Image.open(self.train_image[i]))

        if self.args.dataset == 'wsss4luad':
            background = self.__class__._get_background(image)
            tissue = np.zeros((image.shape[0], image.shape[1]))
            tissue[background == 255] = 0
            tissue[background == 0] = 127
        else:
            tissue = np.ones((image.shape[0], image.shape[1])) * 127

        sample = self.transforms(image=image, mask=tissue)
        image, tissue = sample['image'], sample['mask']
        
        return {'image': image, 'tissue': tissue, 'name': str(name)}

    def __len__(self):
        return len(self.train_image)

    @staticmethod
    def _get_background(region):
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        binary = np.uint8(binary)    
        dst = morphology.remove_small_objects(binary==255,min_size=50,connectivity=1)
        mask = np.array(dst, dtype=np.uint8)
        mask = mask * 255

        return mask