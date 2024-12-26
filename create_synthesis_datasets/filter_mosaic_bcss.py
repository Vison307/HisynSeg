import os
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import albumentations as albu
import random

from torchvision import transforms
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

import argparse

# %%
parser = argparse.ArgumentParser(description='Create dataset for mosaic training')
parser.add_argument('--run', type=int)
args = parser.parse_args()
run = args.run

train_dir = Path("./data/BCSS-WSSS/training")

# %%
def get_patch_label(file):
    # [a: Tumor (TUM), b: Stroma (STR), c: Lymphocytic infiltrate (LYM), d: Necrosis (NEC)]
    if isinstance(file, Path):
        file = str(file)
    fname = file[:-4]
    label_str = fname.split(']')[0].split('[')[-1]
    label = [int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])]
    return label

train_image_list = sorted(list(train_dir.glob('*.png')))

# %%
only_tum_num = 0
only_str_num = 0
only_lym_num = 0
only_nec_num = 0
single_type_num = 0

only_tum_list = []
only_str_list = []
only_lym_list = []
only_nec_list = []

for train_image in train_image_list:
    big_label = get_patch_label(train_image)
    if np.sum(big_label) == 1:
        single_type_num += 1
        if big_label[0] == 1:
            only_tum_num += 1
            only_tum_list.append(train_image)
        elif big_label[1] == 1:
            only_str_num += 1
            only_str_list.append(train_image)
        elif big_label[2] == 1:
            only_lym_num += 1
            only_lym_list.append(train_image)
        elif big_label[3] == 1:
            only_nec_num += 1
            only_nec_list.append(train_image)
            
assert single_type_num == only_tum_num + only_str_num + only_lym_num + only_nec_num
assert len(only_tum_list) == only_tum_num
assert len(only_str_list) == only_str_num
assert len(only_lym_list) == only_lym_num
assert len(only_nec_list) == only_nec_num


# %%
def visualize(save=None, **images):
    """PLot images in one row."""
    fontsize=14
    def axarr_show(axarr, image, name):
        if isinstance(image, torch.Tensor):
            if image.ndim == 3: image = image.permute(1, 2, 0)
            if image.is_cuda: image = image.detach().cpu().numpy()
        if name == 'mask': 
            palette = [0, 64, 128, 64, 128, 0, 243, 152, 0, 255, 255, 255] + [0] * 252 * 3
            image = Image.fromarray(np.uint8(image), mode='P')
            image.putpalette(palette)
            axarr.imshow(image)
            axarr.set_title(name, fontsize=fontsize)
        elif 'background' in name:
            palette = [255, 255, 255, 0, 0, 0]
            image = Image.fromarray(np.uint8(image), mode='P')
            image.putpalette(palette)
            axarr.imshow(image)
            axarr.set_title(name, fontsize=fontsize)
        else:
            axarr.imshow(image)
            axarr.set_title(name, fontsize=fontsize)
    n = len(images)
    fig, axarr = plt.subplots(nrows=1, ncols=n, figsize=(8, 8))
    if n == 1:
        name, image = list(images.items())[0]
        axarr_show(axarr, image, name)
        axarr.set_yticks([])
        axarr.set_xticks([])
    else:
        for i, (name, image) in enumerate(images.items()):
            axarr_show(axarr[i], image, name)
            
        for ax in axarr.ravel():
            ax.set_yticks([])
            ax.set_xticks([])
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()
    plt.close()

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
            tile_name = np.random.choice(only_tum_list + only_str_list + only_lym_list + only_nec_list)
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
discriminator.load_state_dict(torch.load(f'./create_synthesis_datasets/weights/dis_bcss_mosaic_2_112_r18_e3_run{run}.pth'))

discriminator = discriminator.cuda()
discriminator.eval();

# %%
patch_num = 2
patch_size = 112

cnt = 0
N = 7200

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

image_save_dir = f"./data/BCSS-WSSS/mosaic_2_112_run{run}/disc_img_r18_e5"
mask_save_dir = f"./data/BCSS-WSSS/mosaic_2_112_run{run}/disc_mask_r18_e5"

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
dataloader = DataLoader(dataset, batch_size=128, num_workers=16, shuffle=False, pin_memory=False)
dataloader = iter(dataloader)

# %%
print('begin generation...')
i = 1
exist_idx = [int(name.split('-')[0]) for name in os.listdir(image_save_dir)]
total_time = 0
while cnt < N:
    # image, mask = synthesize_once(patch_num, patch_size)
    # input = transform(image).unsqueeze(0).cuda()
    start_time = time.time()
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
        while (cnt + 1) in exist_idx:
            cnt += 1
            if cnt >= N:
                break
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
        palette[0:3] = [255, 0, 0]
        palette[3:6] = [0,255,0]
        palette[6:9] = [0,0,255]
        palette[9:12] = [153, 0, 255]
        palette[12:15] = [255, 255, 255]
        mask = Image.fromarray(mask)
        mask.putpalette(palette)
        mask.save(os.path.join(mask_save_dir, f"{cnt}-{label}.png"))

        if cnt >= N:
            break
    end_time = time.time()
    total_time += end_time - start_time
    sys.stdout.write(f"Progressed {i}-{i+len(input_batch)}; Used time: {end_time - start_time:.2f}s; Total time: {total_time:.2f}s; Remain time: {total_time/cnt * N - total_time} \r")
    i += len(input_batch)

# %%



