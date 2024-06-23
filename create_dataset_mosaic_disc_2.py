import os
import torch
from torch.utils.data import Dataset, DataLoader
cpu_num = '8'
os.environ['OMP_NUM_THREADS'] = cpu_num
os.environ['OPENBLAS_NUM_THREADS'] = cpu_num
os.environ['MKL_NUM_THREADS'] = cpu_num
os.environ['VECLIB_MAXIMUM_THREADS'] = cpu_num
os.environ['NUMEXPR_NUM_THREADS'] = cpu_num
torch.set_num_threads(int(cpu_num))
import sys
import utils
import albumentations as albu
import random

import timm
from torchvision import transforms
from pathlib import Path

import numpy as np
from PIL import Image

import argparse
from concurrent.futures import ThreadPoolExecutor
import time
from linetimer import CodeTimer, linetimer

# np.random.seed(42)
# random.seed(42)

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

# @linetimer()
def create_gridded(patch_num, patch_size):      
    H = W = patch_num * patch_size
    image = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)

    crop_fn = albu.Compose([
        albu.PadIfNeeded(min_height=patch_size, min_width=patch_size),
        albu.RandomCrop(width=patch_size, height=patch_size),
    ])

    for i in range(patch_num):
        for j in range(patch_num):
            while(True):
                tile_name = np.random.choice(normal_images + stroma_images + tumor_images)
                assert sum(utils.to_list(utils.get_label(tile_name))) == 1

                tile = np.array(Image.open(tile_name))
                label = utils.to_list(utils.get_label(tile_name)).index(1)
                tile_mask = np.full((tile.shape[0], tile.shape[1]), label)
                background_mask = np.array(Image.open(Path(train_dir) / 'background-mask' / Path(tile_name).name))
                tile_mask[background_mask > 0] = 3

                sample = crop_fn(image=tile, mask=tile_mask)
                tile = sample['image']
                tile_mask = sample['mask']

                # Background area is smaller than 80%
                if np.sum(tile_mask[tile_mask == 3]) < patch_size * patch_size * 0.8:
                    break
            
            image[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size] = tile
            mask[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size] = tile_mask

    return image, mask

# @linetimer()
def create_mosaic(H, W, image_1, mask_1, image_2, mask_2, image_3, mask_3, image_4, mask_4):
    def get_transforms(height, width, p=0.5):
        _transform = [
            albu.Flip(p=p),
            albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=p),
            albu.RandomCrop(height, width),
        ]
        return albu.Compose(_transform)
    
    image = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    # boarder_mask = np.zeros((H, W), dtype=np.uint8)

    h, w = int(H * (random.random()*0.6+0.2)), int(W * (random.random()*0.6+0.2))
    h += h % 2
    w += w % 2

    transform_1 = get_transforms(height=h, width=w, p=0.8)
    sample = transform_1(image=image_1, mask=mask_1)
    image_1, mask_1 = sample['image'], sample['mask']

    transform_2 = get_transforms(height=h, width=W-w, p=0.8)
    sample = transform_2(image=image_2, mask=mask_2)
    image_2, mask_2 = sample['image'], sample['mask']

    transform_3 = get_transforms(height=H-h, width=w, p=0.8)
    sample = transform_3(image=image_3, mask=mask_3)
    image_3, mask_3 = sample['image'], sample['mask']

    transform_4 = get_transforms(height=H-h, width=W-w, p=0.8)
    sample = transform_4(image=image_4, mask=mask_4)
    image_4, mask_4 = sample['image'], sample['mask']

    # boarder_mask[h-2:h+3, 0:W] = 1
    # boarder_mask[0:H, w-2:w+3] = 1
    
    image[:h, :w, :] = image_1
    image[:h, w:W, :] = image_2
    image[h:H, :w, :] = image_3
    image[h:H, w:W, :] = image_4

    # image_blur = cv2.GaussianBlur(image, (11, 11), 0)
    # image[boarder_mask > 0] = image_blur[boarder_mask > 0]

    mask[:h, :w] = mask_1
    mask[:h, w:W] = mask_2
    mask[h:H, :w] = mask_3
    mask[h:H, w:W] = mask_4
    
    return image, mask

# @linetimer()
def synthesize_once(patch_num, patch_size):
    H = W = patch_num * patch_size 
    
    # with CodeTimer("4 create_gridded"):
        # (image_1, mask_1), (image_2, mask_2), (image_3, mask_3), (image_4, mask_4) = [create_gridded(patch_num, patch_size) for _ in range(4)] # [H, W, C]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(create_gridded, [patch_num]*4, [patch_size]*4)
        results_list = list(results)

    (image_1, mask_1), (image_2, mask_2), (image_3, mask_3), (image_4, mask_4) = results_list
            
    image, mask = create_mosaic(H, W, image_1, mask_1, image_2, mask_2, image_3, mask_3, image_4, mask_4)
    # while True:
    #     try:
            
    #         (image_1, mask_1), (image_2, mask_2), (image_3, mask_3), (image_4, mask_4) = [create_gridded(patch_num, patch_size) for _ in range(4)] # [H, W, C]
    #         image, mask = create_mosaic(H, W, image_1, mask_1, image_2, mask_2, image_3, mask_3, image_4, mask_4)
    #         break            
    #     except AssertionError as e:
    #         print(e)
    return image, mask


class SynthesizeDataset(Dataset):
    def __init__(self, patch_num, patch_size, transform=None):
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return 10_000_000_000

    def __getitem__(self, idx):
        
        image, mask = synthesize_once(self.patch_num, self.patch_size)
        if self.transform:
            transform_image = self.transform(image)
        return image, transform_image, mask


def main(discriminator, dataloader, image_save_dir, mask_save_dir, cnt, N):
    i = 1
    exist_idx = [int(name.split('-')[0]) for name in os.listdir(image_save_dir)]
    begin_time = time.time()
    while cnt < N:
        start_time = time.time()
        # with CodeTimer("1 next"):
        image_batch, input_batch, mask_batch = next(dataloader)
        image_batch = image_batch.numpy()
        mask_batch = mask_batch.numpy()

        with torch.no_grad():
            real_prob = torch.softmax(discriminator(input_batch.cuda()), dim=-1)[:, 1]

        real_idx = (real_prob > 0.5).cpu().numpy()
        image_batch, mask_batch = image_batch[real_idx], mask_batch[real_idx]
        real_prob = real_prob[real_idx]
        
    
        for image, mask, prob in zip(image_batch, mask_batch, real_prob):
            while (cnt+1) in exist_idx:
                cnt += 1
                if cnt >= N: break
            cnt += 1
            print(f"[{cnt}/{N}]; Real prob: {prob}")

            categories = np.unique(mask)
            label = [0, 0, 0]
            for category in categories:
                if category < 3:
                    label[category] = 1
            
            image = Image.fromarray(image)
            image.save(os.path.join(image_save_dir, f"{cnt}-{label}.png"))
            palette = [
                0, 64, 128,  # r, g, b for Tumor
                64, 128, 0,  # r, g, b for Stroma
                243, 152, 0,  # r, g, b for Normal
                255, 255, 255,  # r, g, b for background
            ] + [0] * 252 * 3
            mask = Image.fromarray(mask)
            mask.putpalette(palette)
            mask.save(os.path.join(mask_save_dir, f"{cnt}-{label}.png"))

            if cnt >= N:
                break
        end_time = time.time()
        sys.stdout.write(f"Progressed {i}-{i+len(input_batch)}; Batch time: {end_time-start_time:.2f}s; Total time: {end_time-begin_time:.2f}s\r")
        i += len(input_batch)


def parse_args():
    parser = argparse.ArgumentParser(description='Create dataset for mosaic training')
    parser.add_argument('--idx', type=int, default=0, help='index of the dataset, range [0, 1]', choices=range(2))
    parser.add_argument('--run', type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    run = args.run

    train_dir = "/home/fzj/tmpfs/WSSS4LUAD/1.training"
    train_images = list(Path(train_dir).glob('*.png'))
    tumor_images, stroma_images, normal_images = create_data(train_dir)

    image_save_dir = f"./data/WSSS4LUAD/mosaic_2_112_run{run}/disc_img_r18_e5_2"
    mask_save_dir = f"./data/WSSS4LUAD/mosaic_2_112_run{run}/disc_mask_r18_e5_2"

    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    if not os.path.exists(mask_save_dir):
        os.makedirs(mask_save_dir)

    discriminator = timm.create_model('resnet18', pretrained=False, num_classes=2)
    discriminator.load_state_dict(torch.load(f'./weights_2/dis_mosaic_2_112_r18_e5_run{run}.pth', map_location='cpu'))

    discriminator = discriminator.cuda()
    discriminator.eval()

    # 7, 32 for wsss4luad
    # 2, 112 for bcss
    patch_num = 2
    patch_size = 112
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
    ])
    dataset = SynthesizeDataset(patch_num, patch_size, transform)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False, pin_memory=False)
    dataloader = iter(dataloader)

    cnt = 1800 * args.idx
    N = cnt + 1800

    main(discriminator, dataloader, image_save_dir, mask_save_dir, cnt, N)
    


