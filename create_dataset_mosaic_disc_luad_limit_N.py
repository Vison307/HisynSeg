# %% [markdown]
# # README
# Run this notebook to create the dataset for mosaic training.

# %%
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

from torchvision import transforms
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import argparse

# %%
parser = argparse.ArgumentParser(description='Create dataset for mosaic training')
parser.add_argument('--idx', type=int, default=0, help='index of the dataset, range [0, 1]', choices=range(6))
parser.add_argument('--run', type=int)
parser.add_argument('--N_sample', type=int, default=10)
args = parser.parse_args()
run = args.run
N_sample = args.N_sample

# np.random.seed(42)
# random.seed(42)
train_dir = Path(f"./data/LUAD-HistoSeg/limit_N/one_label_N{N_sample}_run{run}")
# train_dir = Path(f"/home/fzj/tmpfs/one_label_N{N_sample}_run{run}")

# %%
def get_patch_label(filename):
    filename = str(filename)
    label_str = '[' + filename.split('[')[-1].split(']')[0] + ']'
    if ',' not in label_str:
        if ' ' in label_str:
            # [0 0 0 1]
            label_str = label_str.replace(' ', ',')
        else:
            # [0001]
            label_str = str([int(i) for i in label_str[1:-1]])
    label = eval(label_str)
    return label

train_image_list = sorted(list(train_dir.glob('*.png')))

# %%
only_tum_num = 0
only_nec_num = 0
only_lym_num = 0
only_tas_num = 0
single_type_num = 0

only_tum_list = []
only_nec_list = []
only_lym_list = []
only_tas_list = []

for train_image in train_image_list:
    big_label = get_patch_label(train_image)
    if np.sum(big_label) == 1:
        single_type_num += 1
        if big_label[0] == 1:
            only_tum_num += 1
            only_tum_list.append(train_image)
        elif big_label[1] == 1:
            only_nec_num += 1
            only_nec_list.append(train_image)
        elif big_label[2] == 1:
            only_lym_num += 1
            only_lym_list.append(train_image)
        elif big_label[3] == 1:
            only_tas_num += 1
            only_tas_list.append(train_image)

print(f"only_tum_num: {only_tum_num}")
print(f"only_nec_num: {only_nec_num}")
print(f"only_lym_num: {only_lym_num}")
print(f"only_tas_num: {only_tas_num}")
print(f"single_type_num: {single_type_num}")

assert single_type_num == only_tum_num + only_nec_num + only_lym_num + only_tas_num
assert len(only_tum_list) == only_tum_num
assert len(only_nec_list) == only_nec_num
assert len(only_lym_list) == only_lym_num
assert len(only_tas_list) == only_tas_num

# %%

def create_gridded(patch_num, patch_size):      
    H = W = patch_num * patch_size
    image = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)

    crop_fn = albu.Compose([
        # albu.Resize(self.patch_num*self.patch_size, self.patch_num*self.patch_size),
        albu.PadIfNeeded(min_height=patch_size, min_width=patch_size),
        albu.RandomCrop(width=patch_size, height=patch_size),
    ])

    for i in range(patch_num):
        for j in range(patch_num):
            tile_name = np.random.choice(only_tum_list + only_nec_list + only_lym_list + only_tas_list)
            assert sum(get_patch_label(tile_name)) == 1

            tile = np.array(Image.open(tile_name))
            label = get_patch_label(tile_name).index(1)
            tile_mask = np.full((tile.shape[0], tile.shape[1]), label)

            sample = crop_fn(image=tile, mask=tile_mask)
            tile = sample['image']
            tile_mask = sample['mask']

            image[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size] = tile
            mask[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size] = tile_mask

    return image, mask

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

def synthesize_once(patch_num, patch_size):
    H = W = patch_num * patch_size 
    while True:
        try:
            (image_1, mask_1), (image_2, mask_2), (image_3, mask_3), (image_4, mask_4) = [create_gridded(patch_num, patch_size) for _ in range(4)] # [H, W, C]
            image, mask = create_mosaic(H, W, image_1, mask_1, image_2, mask_2, image_3, mask_3, image_4, mask_4)
            break            
        except AssertionError as e:
            print(e)
    return image, mask
    

# %%
import timm
discriminator = timm.create_model('resnet18', pretrained=False, num_classes=2)
discriminator.load_state_dict(torch.load(f'./weights_limit_N/dis_luad_mosaic_2_112_N_{N_sample}_r18_e5_run{run}.pth'))

discriminator = discriminator.cuda()
discriminator.eval();

# %%
# 7, 32 for wsss4luad
# 2, 112 for bcss
patch_num = 2
patch_size = 112

cnt = 1800 * args.idx
N = cnt + 1800

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

image_save_dir = f"./data/LUAD-HistoSeg/limit_N/mosaic_2_112_N_{N_sample}_run{run}/disc_img_r18_e5"
mask_save_dir = f"./data/LUAD-HistoSeg/limit_N/mosaic_2_112_N_{N_sample}_run{run}/disc_mask_r18_e5"

if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)
if not os.path.exists(mask_save_dir):
    os.makedirs(mask_save_dir)

# %%
class SynthesizeDataset(Dataset):
    def __init__(self, patch_num, patch_size, transform=None):
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return 10_000_000

    def __getitem__(self, idx):
        image, mask = synthesize_once(self.patch_num, self.patch_size)
        if self.transform:
            transform_image = self.transform(image)
        return image, transform_image, mask

dataset = SynthesizeDataset(patch_num, patch_size, transform)
dataloader = DataLoader(dataset, batch_size=512, num_workers=8, shuffle=False, pin_memory=False)
dataloader = iter(dataloader)

# %%
print('begin generation...')
i = 1
while cnt < N:
    # image, mask = synthesize_once(patch_num, patch_size)
    # input = transform(image).unsqueeze(0).cuda()
    image_batch, input_batch, mask_batch = next(dataloader)
    image_batch = image_batch.numpy()
    mask_batch = mask_batch.numpy()

    with torch.no_grad():
        # real_prob = torch.sigmoid(discriminator(input)).item()
        real_prob = torch.softmax(discriminator(input_batch.cuda()), dim=-1)[:, 1]

    real_idx = (real_prob > 0.5).cpu().numpy()
    image_batch, mask_batch = image_batch[real_idx], mask_batch[real_idx]
    real_prob = real_prob[real_idx]
    
    for image, mask, prob in zip(image_batch, mask_batch, real_prob):
        cnt += 1
        print(f"[{cnt}/{N}]; Real prob: {prob}")

        categories = np.unique(mask)
        label = [0, 0, 0, 0]
        for category in categories:
            if category < 4:
                label[category] = 1
        
        image = Image.fromarray(image)
        image.save(os.path.join(image_save_dir, f"{cnt}-{label}.png"))
        palette = [0]*15
        palette[0:3] = [205,51,51]          # Tumor epithelial (TE)
        palette[3:6] = [0,255,0]            # Necrosis (NEC)
        palette[6:9] = [65,105,225]         # Lymphocyte (LYM)
        palette[9:12] = [255,165,0]         # Tumor-associated stroma (TAS)
        palette[12:15] = [255, 255, 255]    # White background or exclude
        mask = Image.fromarray(mask)
        mask.putpalette(palette)
        mask.save(os.path.join(mask_save_dir, f"{cnt}-{label}.png"))

        if cnt >= N:
            break
    
    sys.stdout.write(f"Progressed {i}-{i+len(input_batch)}\r")
    i += len(input_batch)

# %%



