import argparse
import logging

import ttach as tta
from loss import mIoUMask
from models.mosaic_module_v2 import MosaicModule

import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from dataset import *
import os

# from models.segmentation_module import SegmentationModule

cpu_num = '2'
os.environ['OMP_NUM_THREADS'] = cpu_num
os.environ['OPENBLAS_NUM_THREADS'] = cpu_num
os.environ['MKL_NUM_THREADS'] = cpu_num
os.environ['VECLIB_MAXIMUM_THREADS'] = cpu_num
os.environ['NUMEXPR_NUM_THREADS'] = cpu_num
torch.set_num_threads(int(cpu_num))


def parse_args():

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
        return self.model(data=x)['logits']

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
    # model = Adaptor(model)
    model = tta.SegmentationTTAWrapper(Adaptor(model), tta.aliases.d4_transform(), merge_mode='mean')

    if isinstance(model, tta.SegmentationTTAWrapper):
        args.save_dir += '_tta'

    if args.dataset == 'wsss4luad':
        test_iou = mIoUMask(num_classes=3)
    else:
        test_iou = mIoUMask(num_classes=4)

    pred_big_mask_dict_ms = dict()
    cnt_big_mask_dict_ms = dict()
    pred_big_mask_dict = dict()
    cnt_big_mask_dict = dict()

    print(f'Save dir: {args.save_dir}')
    if not os.path.exists(os.path.join(args.save_dir, 'mask')):
        os.makedirs(os.path.join(args.save_dir, 'mask'))
    if not os.path.exists(os.path.join(args.save_dir, 'prob')):
        os.makedirs(os.path.join(args.save_dir, 'prob'))

    if args.dataset in ['bcss', 'luad']:
        output_patches = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            image_batch, mask_batch, name_batch, original_h_batch, original_w_batch = data
            if i % 100 == 0:
                print(f'{i}/{len(test_dataloader)}')
                print(f'mIoU(patch): {test_iou.Mean_Intersection_over_Union()}')
                print(f'fwIoU: {test_iou.Frequency_Weighted_Intersection_over_Union()}')
                if args.dataset == 'wsss4luad':
                    print(f'tIoU, sIoU, nIoU: {test_iou.Tissue_Intersection_over_Union()}')
                elif args.dataset == 'bcss':
                    print(f'tIoU, sIoU, lIoU, nIoU: {test_iou.Tissue_Intersection_over_Union()}')
                elif args.dataset == 'luad':
                    print(f'tIoU, nIoU, lIoU, sIoU: {test_iou.Tissue_Intersection_over_Union()}')
            image_batch = image_batch.cuda()
            mask_batch = mask_batch.cuda()
            output = model(image_batch)
            # print(f'output.shape: {output.shape}; mask_batch.shape: {mask_batch.shape}')
            test_iou(output, mask_batch)

            if args.dataset == 'wsss4luad':
                for j in range(image_batch.shape[0]):
                    original_w, original_h = original_w_batch[j], original_h_batch[j]

                    output_ = output[j][:, :original_h, :original_w] # logits, C x 256 x 256

                    mask_ = mask_batch[j][:original_h, :original_w].cpu().numpy() # mask, 256 x 256
                    
                    # [C, H, W] x [C, 1, 1]
                    probs = torch.softmax(output_, dim=0) # * class_pred[j].unsqueeze(1).unsqueeze(2)
                    probs = probs.cpu().numpy()

                    probs = probs.transpose(1, 2, 0) # [H, W, C]

                    name = name_batch[j]
                    # patch_label = utils.to_list(name[-13:-4])
                    tissue = np.zeros_like(mask_, dtype=np.uint8)
                    tissue[mask_ != 3] = 1

                    image_idx = name.split('_')[0]
                    scale = float(name.split('_')[1])
                    position = (int(name.split('_')[2]), int(name.split('_')[3].split('-')[0]))

                    dict_key = f'{image_idx}_{scale}'

                    if dict_key not in pred_big_mask_dict_ms:
                        w, h = Image.open(os.path.join('/'.join(args.test_data.split('/')[:-1]), 'img', image_idx + '.png')).size
                        w_ = int(w * scale)
                        h_ = int(h * scale)
                        pred_big_mask_dict_ms[dict_key] = np.zeros((h_, w_, 3))
                        cnt_big_mask_dict_ms[dict_key] = np.zeros((h_, w_, 1))
                        
                    pred_big_mask_dict_ms[dict_key][position[0]:position[0]+output_.shape[1], position[1]:position[1]+output_.shape[2], :] += probs
                    cnt_big_mask_dict_ms[dict_key][position[0]:position[0]+output_.shape[1], position[1]:position[1]+output_.shape[2], :] += 1

                    del probs
                    del output_
                    del mask_
                    del tissue
            else:
                for output_, name in zip(output, name_batch):
                    pred_mask = torch.argmax(output_, dim=0).cpu().numpy()
                    output_patches.append((name, pred_mask))

    if args.dataset == 'wsss4luad':
        # ----> All validation patches are predicted, now we can calculate the final miou
        for k, mask in pred_big_mask_dict_ms.items():
            mask /= cnt_big_mask_dict_ms[k] # [H, W, 3]
            image_idx = k.split('_')[0]
            scale = k.split('_')[1]

            if image_idx not in pred_big_mask_dict:
                w, h = Image.open(os.path.join('/'.join(args.test_data.split('/')[:-1]), 'img', image_idx + '.png')).size
                pred_big_mask_dict[image_idx] = np.zeros((h, w, 3))
                cnt_big_mask_dict[image_idx] = np.zeros((h, w, 1))
            else:
                h, w = pred_big_mask_dict[image_idx].shape[:2]

            mask = interpolate_tensor(torch.from_numpy(mask.transpose(2, 0, 1)), (h, w)).numpy().transpose(1, 2, 0)
            pred_big_mask_dict[image_idx][:, :, :] += mask
            cnt_big_mask_dict[image_idx][:, :, :] += 1

    if args.dataset == 'wsss4luad':
        big_mask_iou = mIoUMask()
        for k, mask_pred in pred_big_mask_dict.items():
            mask_pred /= cnt_big_mask_dict[k]
            mask = np.array(Image.open(os.path.join('/'.join(args.test_data.split('/')[:-1]), 'mask', k + '.png')))
            big_mask_iou(torch.from_numpy(mask_pred.transpose(2, 0, 1)).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0), probs=True)
            
            mask_pred = np.argmax(mask_pred, axis=2)
            # background positions are known
            mask_pred[mask == 3] = 3
            palette = [0, 64, 128, 64, 128, 0, 243, 152, 0, 255, 255, 255] + [0] * 252 * 3
            mask_pred = Image.fromarray(np.uint8(mask_pred), mode='P')
            mask_pred.putpalette(palette)
            mask_pred.save(os.path.join(args.save_dir, 'mask', k + '.png'))

        logging.critical(f'Segmentation Test - Test mIoU (patch): {test_iou.Mean_Intersection_over_Union()}')
        logging.critical(f'Segmentation Test - Test fwIoU (patch): {test_iou.Frequency_Weighted_Intersection_over_Union()}')
        logging.critical(f'Segmentation Test - Test tissue IoU (patch): {test_iou.Tissue_Intersection_over_Union()}')

        print(f'mIoU(big mask): {big_mask_iou.Mean_Intersection_over_Union()}')
        print(f'fwIoU: {big_mask_iou.Frequency_Weighted_Intersection_over_Union()}')
        print(f'tIoU, sIoU, nIoU: {big_mask_iou.Tissue_Intersection_over_Union()}')

        logging.critical(f'Segmentation Test - Test mIoU (big mask): {big_mask_iou.Mean_Intersection_over_Union()}')
        logging.critical(f'Segmentation Test - Test fwIoU (big mask): {big_mask_iou.Frequency_Weighted_Intersection_over_Union()}')
        logging.critical(f'MosaSegmentationic Test - Test tissue IoU (big mask): {big_mask_iou.Tissue_Intersection_over_Union()}')
    else:
        for name, output_mask in output_patches:
            mask = np.array(Image.open(os.path.join(args.test_data, 'mask', name)))
            assert output_mask.shape[0] == mask.shape[0] and output_mask.shape[1] == mask.shape[1]

            if args.dataset == 'bcss':
                palette = [0]*15
                palette[0:3] = [255, 0, 0]
                palette[3:6] = [0,255,0]
                palette[6:9] = [0,0,255]
                palette[9:12] = [153, 0, 255]
                palette[12:15] = [255, 255, 255]
            elif args.dataset == 'luad':
                palette = [0]*15
                palette[0:3] = [205,51,51]          # Tumor epithelial (TE)
                palette[3:6] = [0,255,0]            # Necrosis (NEC)
                palette[6:9] = [65,105,225]         # Lymphocyte (LYM)
                palette[9:12] = [255,165,0]         # Tumor-associated stroma (TAS)
                palette[12:15] = [255, 255, 255]    # White background or exclude

            output_mask = Image.fromarray(np.uint8(output_mask), mode='P')
            output_mask.putpalette(palette)
            output_mask.save(os.path.join(args.save_dir, 'mask', name))

        print(f'mIoU(big mask): {test_iou.Mean_Intersection_over_Union()}')
        print(f'fwIoU: {test_iou.Frequency_Weighted_Intersection_over_Union()}')
        if args.dataset == 'bcss':
            print(f'tmr, str, lym, nec: {test_iou.Tissue_Intersection_over_Union()}')
        elif args.dataset == 'luad':
            print(f'tmr, nec, lym, str: {test_iou.Tissue_Intersection_over_Union()}')

        logging.critical(f'Segmentation Test - Test mIoU (big mask): {test_iou.Mean_Intersection_over_Union()}')
        logging.critical(f'Segmentation Test - Test fwIoU (big mask): {test_iou.Frequency_Weighted_Intersection_over_Union()}')
        logging.critical(f'Segmentation Test - Test tissue IoU (big mask): {test_iou.Tissue_Intersection_over_Union()}')

if __name__ == '__main__':
    args = parse_args()
    # pl.seed_everything(42)

    if os.path.isfile(args.checkpoint):
        args.save_dir = os.path.join(os.path.dirname(args.checkpoint), 'test')
    else:
        args.save_dir = os.path.join(args.checkpoint, 'test')

    logging.basicConfig(
        level=logging.CRITICAL, 
        filename=f"{os.path.dirname(args.save_dir)}/segmentation_test.log", 
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
