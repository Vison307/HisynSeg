import os
import torch
cpu_num = '4'
os.environ['OMP_NUM_THREADS'] = cpu_num
os.environ['OPENBLAS_NUM_THREADS'] = cpu_num
os.environ['MKL_NUM_THREADS'] = cpu_num
os.environ['VECLIB_MAXIMUM_THREADS'] = cpu_num
os.environ['NUMEXPR_NUM_THREADS'] = cpu_num
torch.set_num_threads(int(cpu_num))

import argparse
import logging
from pathlib import Path
import random

import numpy as np
from PIL import Image
import ttach as tta
from loss import mIoUMask
from models.mosaic_module_v2 import MosaicModule

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics
from tqdm import tqdm

def get_label(dataset, filename):
    label_str = filename.split('[')[-1].split(']')[0]
    if dataset == 'wsss4luad':
        label = [int(label_str[0]), int(label_str[3]), int(label_str[6])]
    elif dataset == 'bcss':
        label = [int(label_str[0]), int(label_str[1]), int(label_str[2]), int(label_str[3])]
    elif dataset == 'luad':
        label = [int(label_str[0]), int(label_str[2]), int(label_str[4]), int(label_str[6])]
    return label

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

    def __len__(self):
        return len(self.test_image)

    def __getitem__(self, i):
        name = self.test_image[i].name
        label = get_label(self.args.dataset, name)
        image = self.test_image[i]
        image = np.array(Image.open(image))
        mask = np.array(Image.open(self.test_data / 'mask' / name))
        original_h, original_w = image.shape[:2]

        sample = self.transforms(image=image, mask=mask)
        image, mask = sample['image'], sample['mask'].long()

        return image, torch.Tensor(label).long(), mask, name, original_h, original_w


def parse_args():
    # CUDA_VISIBLE_DEVICES=7, python segmentation_test_class.py -ckpt logs/wsss4luad3k6_3k6 --gpus=0, --patch-size=224 --test-data data/WSSS4LUAD/3.testing/patches_224_112 --dataset wsss4luad

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='wsss4luad')

    parser.add_argument('--checkpoint', '-ckpt', help='path to the checkpoint file')

    parser.add_argument('--patch-size', type=int, default=256)

    parser.add_argument('--test-data', default='./data/testing')
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--gpus', default=[1,])
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--pin-memory', action='store_true', default=True)

    args = parser.parse_args()

    return args

def interpolate_tensor(tensor, target_shape):
    return F.interpolate(tensor.unsqueeze(0), target_shape, mode='bilinear')[0]

class Adaptor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(data=x)['cls_logits']

def main(args):
    # ----> Testing with the best model in mIoU

    lib = torch.load(args.checkpoint, map_location='cpu')
    model_args = lib['hyper_parameters']['args']
    for k, v in vars(args).items():
        setattr(model_args, k, v)
    args = model_args
    if 'w1' not in args:
        args.w1 = 1.0
        args.w2 = 1.0
        args.w3 = 1.0
    
    test_dataset = TestDataset(args)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=False)

    model = MosaicModule.load_from_checkpoint(args.checkpoint, args=args)
    model = model.cuda()
    model = Adaptor(model)

    if args.dataset == 'wsss4luad':
        test_metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(num_classes=3, average='micro'),
            torchmetrics.F1Score(num_classes=3, average='macro'),
            torchmetrics.Precision(num_classes=3, average='macro'),
            torchmetrics.Recall(num_classes=3, average='macro'),
            torchmetrics.AUROC(num_classes=3, average='macro')
        ])
    else:
        test_metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(num_classes=4, average='micro'),
            torchmetrics.F1Score(num_classes=4, average='macro'),
            torchmetrics.Precision(num_classes=4, average='macro'),
            torchmetrics.Recall(num_classes=4, average='macro'),
            torchmetrics.AUROC(num_classes=4, average='macro')
        ])
    test_metrics = test_metrics.cuda()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            image_batch, label_batch, mask_batch, name_batch, original_h_batch, original_w_batch = data
            image_batch = image_batch.cuda()
            label_batch = label_batch.cuda()
            output = model(image_batch)
            # print(f'output.shape: {output.shape}; mask_batch.shape: {mask_batch.shape}')
            metrics = test_metrics(output, label_batch)
            print(f'batch {i+1}/{len(test_dataloader)}: {metrics}')
            
    accuracy = test_metrics['Accuracy'].compute()
    f1_score = test_metrics['F1Score'].compute()
    precision = test_metrics['Precision'].compute()
    recall = test_metrics['Recall'].compute()
    auroc = test_metrics['AUROC'].compute()
    logging.critical(f'Classification Test - Test Accuracy: {accuracy}')
    logging.critical(f'Classification Test - Test F1 Score: {f1_score}')
    logging.critical(f'Classification Test - Test Precision: {precision}')
    logging.critical(f'Classification Test - Test Recall: {recall}')
    logging.critical(f'Classification Test - Test AUROC: {auroc}')

if __name__ == '__main__':
    args = parse_args()
    # pl.seed_everything(42)

    if os.path.isfile(args.checkpoint):
        args.save_dir = os.path.join(os.path.dirname(args.checkpoint), 'test')
    else:
        args.save_dir = os.path.join(args.checkpoint, 'test')

    logging.basicConfig(
        level=logging.CRITICAL, 
        filename=f"{os.path.dirname(args.save_dir)}/classification_test.log", 
        filemode='a',
        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s - %(funcName)s",
        datefmt="%Y-%m-%d %H:%M:%S" 
    )
    logging.critical(args)

    if args.checkpoint.endswith('.ckpt'):
        checkpoint_file_path = args.checkpoint
    else:
        max_score = 0.0
        checkpoint_file_path = None
        for filename in os.listdir(args.checkpoint):
            if 'epoch=' in filename:
                score = float(filename.split('=')[-1].split('.ckpt')[0])
                if score > max_score:
                    max_score = score
                    checkpoint_file_path = os.path.join(args.checkpoint, filename)
                    
        assert checkpoint_file_path is not None, f'Cannot find a valid checkpoint file in {args.checkpoint}'

    args.checkpoint = checkpoint_file_path
    logging.critical(f'Find best checkpoint file: {args.checkpoint}')
    print(f'Find best checkpoint file: {args.checkpoint}')

    main(args)
