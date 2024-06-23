"""
Stage 1 - Train on dataset with Mosaic
"""
import os
import argparse
import logging
from pathlib import Path
import random
import shutil
import inspect
import numpy as np
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.mosaic_module_v2 import MosaicModule

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset, Subset


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from dataset import LabeledDataset, UnlabeledDataset
from albumentations import ImageOnlyTransform
import stain_transforms

import time
execution_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

cpu_num = '4'
os.environ['OMP_NUM_THREADS'] = cpu_num
os.environ['OPENBLAS_NUM_THREADS'] = cpu_num
os.environ['MKL_NUM_THREADS'] = cpu_num
os.environ['VECLIB_MAXIMUM_THREADS'] = cpu_num
os.environ['NUMEXPR_NUM_THREADS'] = cpu_num
torch.set_num_threads(int(cpu_num))

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# torch.autograd.set_detect_anomaly(True)

def parse_args():
    # python mosaic_train.py --model=UnetPlusPlus --encoder=efficientnet-b6 --lr=0.001 --gpus=1, --epochs=15 --batch-size=16 --mosaic-data=data/BCSS-WSSS/mosaic_2_112 --patch-size=224 --val-data data/BCSS-WSSS/val --num-classes 4 --dataset
    # import os bcss

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--model', help='the model for training')
    parser.add_argument('--encoder', help='the encoder of the model', default='efficientnet-b6')
    
    parser.add_argument('--w1', type=float, default=1.0)
    parser.add_argument('--w2', type=float, default=1.0)
    parser.add_argument('--w3', type=float, default=1.0)

    parser.add_argument('--num-classes', help='categories in prediction mask', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='wsss4luad')

    parser.add_argument('--log_dir', default='mosaic_logs')

    parser.add_argument('--tta', action='store_true', default=False)

    parser.add_argument('--patch-size', type=int, default=224)
    
    parser.add_argument('--semi_image_dir', type=str, default=None)

    parser.add_argument('--train_image_dirs', type=str, nargs='+')
    parser.add_argument('--train_mask_dirs', type=str, nargs='+')
    parser.add_argument('--num_samples', type=int, nargs='+', default=[-1])
    parser.add_argument('--val_data', default='./data/validation')
    parser.add_argument('--test_data', default='./data/testing')

    parser.add_argument('--gpus', default=[1,])
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')

    args = parser.parse_args()

    return args

#---->load Loggers
def load_loggers(args, name):
    args.log_path = Path(args.log_dir)
    (args.log_path / 'code').mkdir(parents=True, exist_ok=True)

    shutil.copyfile(__file__, args.log_path / 'code' / 'mosaic_train_v2.py')
    shutil.copyfile(inspect.getfile(MosaicModule), args.log_path / 'code' / 'mosaic_module_v2.py')
    shutil.copyfile(inspect.getfile(LabeledDataset), args.log_path / 'code' / 'dataset.py')
    shutil.copyfile('models/smp_model.py', args.log_path / 'code' / 'smp_model.py')

    logging.basicConfig(
        level=logging.CRITICAL, 
        filename=os.path.join(args.log_dir, 'train.log'), 
        filemode='w',
        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s - %(funcName)s",
        datefmt="%Y-%m-%d %H:%M:%S" 
    )

    logging.critical(args)

    #---->TensorBoard
    tb_logger = pl_loggers.TensorBoardLogger(
        args.log_dir,
        name='',
        version='', 
        default_hp_metric = False)
        
    #---->CSV
    csv_logger = pl_loggers.CSVLogger(
        args.log_dir,
        name='',
        version='',)
    
    return [tb_logger, csv_logger]

def load_callbacks(args):
    Mycallbacks = []
    Mycallbacks.append(
        ModelCheckpoint(
            monitor='validation_miou_mask_epoch',
            filename='{epoch:02d}-{validation_miou_mask_epoch:.4f}',
            save_last=True,
            verbose = True,
            mode='max',
            dirpath = str(args.log_path),
            save_weights_only=True,
            save_top_k=1,
        )
    )
    Mycallbacks.append(LearningRateMonitor())
            
    return Mycallbacks

    
def main(args, experiment_name):
    # ----> loggers and callbacks
    tb_logger = load_loggers(args, experiment_name)
    callbacks = load_callbacks(args)

    # ----> Build Fully-supervised Train Dataset
    train_dataset = []
    transforms = albu.Compose([
            albu.RandomResizedCrop(height=args.patch_size, width=args.patch_size, scale=(0.9, 1)),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(p=0.5),
            albu.OpticalDistortion(p=0.5),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),  # Normalization
            ToTensorV2(transpose_mask=True),  # [H, W, C] -> [C, H, W]
        ])
    
    if args.num_samples == [-1]:
        args.num_samples = [-1] * len(args.train_image_dirs)
    assert len(args.train_image_dirs) == len(args.train_mask_dirs) == len(args.num_samples), f'len of train_image_dirs: {len(args.train_image_dirs)}, len of train_mask_dirs: {len(args.train_mask_dirs)}, len of num_samples: {len(args.num_samples)} Not Matched'
    
    for train_image_dir, train_mask_dir, num_samples in zip(args.train_image_dirs, args.train_mask_dirs, args.num_samples):
        logging.critical(f'Training set: {train_image_dir} - {train_mask_dir}')
        dataset = LabeledDataset(train_image_dir, train_mask_dir, transforms=transforms, num_classes=args.num_classes) # sort by filename
        if num_samples != -1:
            dataset = Subset(dataset, range(num_samples))
        logging.critical(f'len of Training set: {len(dataset)}')
        train_dataset.append(dataset)
    train_dataset = ConcatDataset(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=8, drop_last=True, pin_memory=False,
    )

    # ----> Build Semi-supervised dataset
    if args.semi_image_dir is not None:
        semi_dataset = UnlabeledDataset(args.semi_image_dir, transforms=transforms, num_classes=args.num_classes)
        semi_train_dataloader = DataLoader(semi_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=False)
        train_dataloaders = {'labeled': train_dataloader, 'unlabeled': semi_train_dataloader}
    else:
        train_dataloaders = train_dataloader

    # ----> Build Validation Dataset
    val_image_dir = os.path.join(args.val_data, 'img')
    val_mask_dir = os.path.join(args.val_data, 'mask')
    val_transform = albu.Compose([
            albu.PadIfNeeded(args.patch_size, args.patch_size, border_mode=2, position=albu.PadIfNeeded.PositionType.TOP_LEFT),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),  # Normalization
            ToTensorV2(transpose_mask=True),  # [H, W, C] -> [C, H, W]
        ])
    val_dataset = LabeledDataset(val_image_dir, val_mask_dir, transforms=val_transform, num_classes=args.num_classes)
    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=8, pin_memory=False)  
      
    # ----> Model
    model = MosaicModule(args)

    # ----> Trainer
    if 'Unet' in args.model:
        trainer = pl.Trainer(
            # precision=16,
            logger=tb_logger, 
            max_epochs=args.epochs, 
            gpus=args.gpus, 
            callbacks=callbacks,
            # deterministic=False,
        )
        # torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        trainer = pl.Trainer(
            precision=16,
            logger=tb_logger, 
            max_epochs=args.epochs, 
            gpus=args.gpus, 
            callbacks=callbacks,
        )
    
    trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloader)
    
    # ----> Testing with the best model in mIoU
    logging.critical(f'best path: {callbacks[0].best_model_path}')
    # args.tta = True # set to True to test with TTA
    # model = MosaicModule.load_from_checkpoint(callbacks[0].best_model_path, args=args)

    # # ----> Testing
    # test_image_dir = os.path.join(args.test_data, 'img')
    # test_mask_dir = os.path.join(args.test_data, 'mask')
    # test_transform = albu.Compose([
    #         albu.PadIfNeeded(args.patch_size, args.patch_size, border_mode=2, position=albu.PadIfNeeded.PositionType.TOP_LEFT),
    #         albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),  # Normalization
    #         ToTensorV2(transpose_mask=True),  # [H, W, C] -> [C, H, W]
    #     ])
    # test_dataset = LabeledDataset(test_image_dir, test_mask_dir, transforms=test_transform, num_classes=args.num_classes)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=8)

    # test_result = trainer.validate(model, test_dataloader)
    # logging.critical(f'Test Result: {test_result}')


def get_experiment_name(args):
    experiment_name = f'{args.model}:{args.encoder}:{args.patch_size}:{args.batch_size}:{args.lr}'
    return experiment_name


if __name__ == '__main__':
    args = parse_args()
    if args.seed:
        pl.seed_everything(args.seed)

    experiment_name = get_experiment_name(args)

    main(args, experiment_name)
