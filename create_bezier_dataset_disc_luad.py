# %% [markdown]
# # README
# Run this notebook to create the dataset for mosaic training.

# %%
import os
cpu_num = '2'
os.environ['OMP_NUM_THREADS'] = cpu_num
os.environ['OPENBLAS_NUM_THREADS'] = cpu_num
os.environ['MKL_NUM_THREADS'] = cpu_num
os.environ['VECLIB_MAXIMUM_THREADS'] = cpu_num
os.environ['NUMEXPR_NUM_THREADS'] = cpu_num
import torch
torch.set_num_threads(int(cpu_num))

import utils
import random
import sys

from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import argparse

# %%
# np.random.seed(42)
# random.seed(42)
parser = argparse.ArgumentParser(description='Create dataset for mosaic training')
parser.add_argument('--run', type=int)
args = parser.parse_args()
run = args.run

# %%
def get_patch_label(filename):
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

def create_data(train_data):
    only_tum_list = []
    only_nec_list = []
    only_lym_list = []
    only_tas_list = []
    train_image_list = os.listdir(train_data)
    for name in train_image_list:
        big_label = get_patch_label(name)
        if np.sum(big_label) == 1:
            train_image = os.path.join(train_data, name)
            if big_label[0] == 1:
                only_tum_list.append(train_image)
            elif big_label[1] == 1:
                only_nec_list.append(train_image)
            elif big_label[2] == 1:
                only_lym_list.append(train_image)
            elif big_label[3] == 1:
                only_tas_list.append(train_image)

    return only_tum_list, only_nec_list, only_lym_list, only_tas_list

# %%
import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt


bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)

# %%
def get_bezier_mask(n, scale, rad=0.2, edgy=0.05):
    a = get_random_points(n=n, scale=scale)
    x, y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)

    x = np.round(x)
    y = np.round(y)
    mask = np.zeros((scale, scale), dtype=np.uint8)
    mask = cv2.fillPoly(mask, np.int32([np.stack([x, y], axis=1)]), 1)

    return mask

def get_onelabel_mask(category, scale):
    if category == "tum":
        return np.zeros((scale, scale), dtype=np.uint8)
    elif category == "nec":
        return np.ones((scale, scale), dtype=np.uint8)
    elif category == "lym":
        return (np.ones((scale, scale), dtype=np.uint8) * 2)
    else:
        return (np.ones((scale, scale), dtype=np.uint8) * 3)


# %%
train_dir = "./data/LUAD-HistoSeg/train/"
only_tum_list, only_nec_list, only_lym_list, only_tas_list = create_data(train_dir)
dataset_dict = {
    "tum": only_tum_list,
    "nec": only_nec_list,
    "lym": only_lym_list,
    'tas': only_tas_list
}

def synthesize_one(n=12, rad=0.2, edgy=0.05, background_class="tum", foreground_class="nec"):
    background_image_path = random.choice(dataset_dict[background_class])
    foreground_image_path = random.choice(dataset_dict[foreground_class])

    background_image = np.array(Image.open(background_image_path).resize((224, 224)))
    foreground_image = np.array(Image.open(foreground_image_path).resize((224, 224)))

    background_mask = get_onelabel_mask(background_class, scale=224)
    foreground_mask = get_onelabel_mask(foreground_class, scale=224)

    bezier_mask = get_bezier_mask(n=n, scale=224, rad=rad, edgy=edgy)

    synthesized_image = bezier_mask[:,:,np.newaxis] * foreground_image + (1 - bezier_mask)[:,:,np.newaxis] * background_image
    synthesized_mask = bezier_mask * foreground_mask + (1 - bezier_mask) * background_mask

    return synthesized_image, synthesized_mask


# %%
save_dir = f"data/LUAD-HistoSeg/bezier224_5_0.2_0.05_1d1_run{run}"

if not os.path.exists(os.path.join(save_dir, 'disc_img_r18_e5')):
    os.makedirs(os.path.join(save_dir, 'disc_img_r18_e5'))
if not os.path.exists(os.path.join(save_dir, 'disc_mask_r18_e5')):
    os.makedirs(os.path.join(save_dir, 'disc_mask_r18_e5'))

# %%
import timm
from torchvision import transforms
discriminator = timm.create_model('resnet18', num_classes=2, pretrained=False)
discriminator.load_state_dict(torch.load(f'weights/dis_luad_bezier224_5_0.2_0.05_1d1_r18_e5_run{run}.pth'))
discriminator = discriminator.cuda()
discriminator.eval();

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

# %%
N_train = 3_600
cnt = 0
i = 1
while cnt < N_train:
    background_class, foreground_class = np.random.choice(['tum', 'nec', 'lym', 'tas'], size=2, replace=False)
    image, mask = synthesize_one(n=5, background_class=background_class, foreground_class=foreground_class)
    input = transform(image).unsqueeze(0).cuda()
    with torch.no_grad():
        # real_prob = torch.sigmoid(discriminator(input)).item()
        real_prob = torch.softmax(discriminator(input), dim=-1)[:, 1].item()
    categories = np.unique(mask)
    label = [0, 0, 0, 0]
    for category in categories:
        if category < 4:
            label[category] = 1
    
    if real_prob > 0.5:
        cnt += 1
        print(f"[{cnt}/{N_train}]; Real prob: {real_prob}")
        image = Image.fromarray(image)
        image.save(os.path.join(save_dir, 'disc_img_r18_e5', f"{cnt}-{label}.png"))
        palette = [0]*15
        palette[0:3] = [205,51,51]          # Tumor epithelial (TE)
        palette[3:6] = [0,255,0]            # Necrosis (NEC)
        palette[6:9] = [65,105,225]         # Lymphocyte (LYM)
        palette[9:12] = [255,165,0]         # Tumor-associated stroma (TAS)
        palette[12:15] = [255, 255, 255]    # White background or exclude
        mask = Image.fromarray(mask)
        mask.putpalette(palette)
        mask.save(os.path.join(save_dir, 'disc_mask_r18_e5', f"{cnt}-{label}.png"))
    else:
        sys.stdout.write(f"Progressing {i}, Real prob: {real_prob}\r")
    
    i += 1
