import os
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import AdamW

import pytorch_lightning as pl

from models.smp_model import SmpModel
from loss import mIoUMask


class MosaicModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        
        self.train_iou = mIoUMask(num_classes=args.num_classes)
        self.valid_iou = mIoUMask(num_classes=args.num_classes)

        w1 = getattr(args, 'w1', 1.0)
        w2 = getattr(args, 'w2', 1.0)
        w3 = getattr(args, 'w3', 1.0)

        self.model = SmpModel(args.model, args.encoder, args.num_classes, w1=w1, w2=w2, w3=w3)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer

    def forward(self, **kwargs):
        y = self.model(**kwargs)
        return y
    
    
    def training_step(self, batch, batch_idx):
        if 'labeled' in batch:
            # fully supervised
            labeled_batch = batch['labeled']
            image, mask, label = labeled_batch['image'], labeled_batch['mask'], labeled_batch['label']
            labeled_result_dict = self(data=image, mask=mask, label=label)

            labeled_loss_seg = labeled_result_dict['loss_seg']
            labeled_loss_cls = labeled_result_dict['loss_cls']
            labeled_loss = labeled_result_dict['loss']

            mask_pred = labeled_result_dict['logits']

            self.log('train_loss_seg', labeled_loss_seg, prog_bar=True)
            self.log('train_loss_labeled_cls', labeled_loss_cls, prog_bar=True)

            self.train_iou(mask_pred, mask)     

            # semi supervised
            unlabeled_batch = batch['unlabeled']
            image, label = unlabeled_batch['image'], unlabeled_batch['label']
            unlabeled_result_dict = self(data=image, label=label)
            unlabeled_loss_cls = unlabeled_result_dict['loss_cls']
            unlabeled_reg = unlabeled_result_dict['reg']
            unlabeled_loss = unlabeled_result_dict['loss']

            self.log('train_loss_reg', unlabeled_reg, prog_bar=True)
            self.log('train_loss_unlabeled_cls', unlabeled_loss_cls, prog_bar=True)
        
            loss = labeled_loss + unlabeled_loss
        else:
            image, mask, label = batch['image'], batch['mask'], batch['label']
            labeled_result_dict = self(data=image, mask=mask, label=label)

            labeled_loss_seg = labeled_result_dict['loss_seg']
            labeled_loss_cls = labeled_result_dict['loss_cls']
            loss = labeled_result_dict['loss']

            mask_pred = labeled_result_dict['logits']

            self.log('train_loss_seg', labeled_loss_seg, prog_bar=True)
            self.log('train_loss_cls', labeled_loss_cls, prog_bar=True)

            self.train_iou(mask_pred, mask)     

        self.log('train_loss', loss, prog_bar=True)     
        self.log('train_miou_epoch', self.train_iou.Mean_Intersection_over_Union(), prog_bar=True)
        return loss
    
    def training_epoch_end(self, training_step_outputs):
        self.log('train_miou_epoch', self.train_iou.Mean_Intersection_over_Union())
        self.log('train_fwiou_epoch', self.train_iou.Frequency_Weighted_Intersection_over_Union())
        self.train_iou.reset()

    def on_validation_epoch_start(self):
        tissue_iou = self.train_iou.Tissue_Intersection_over_Union()
        miou = self.train_iou.Mean_Intersection_over_Union()
        fwiou = self.train_iou.Frequency_Weighted_Intersection_over_Union()
        print('\n' + '-' * 50)
        print("\nExperiment Settings")
        print(f"Labeled Dataset: \033[1;34m{self.args.train_image_dirs}\033[0m")
        print(f"Unlabeled Dataset: \033[1;34m{self.args.semi_image_dir}\033[0m")
        print(f"Log Path: \033[1;34m{self.args.log_path}\033[0m")

        print('\n' + '-' * 50)
        print(f"Training Result")
        
        if self.args.dataset == 'wsss4luad':
            print(f'Tumor IoU: \033[1;35m{tissue_iou[0]:.4f}\033[0m')
            print(f'Stroma IoU: \033[1;35m{tissue_iou[1]:.4f}\033[0m')
            print(f'Normal IoU: \033[1;35m{tissue_iou[2]:.4f}\033[0m')
            print(f'mIoU: \033[1;35m{miou:.4f}\033[0m')
            print(f'fwIoU: \033[1;35m{fwiou:.4f}\033[0m')
            print('\n' + '-' * 50)
            self.pred_big_mask_dict_ms_val = dict()
            self.cnt_big_mask_dict_ms_val = dict()
            self.pred_big_mask_dict_val = dict()
            self.cnt_big_mask_dict_val = dict()

        elif self.args.dataset == 'bcss':
            self.pred_big_mask_dict_ms_val = None
            self.cnt_big_mask_dict_ms_val = None
            print(f'Tumor IoU: \033[1;35m{tissue_iou[0]:.4f}\033[0m')
            print(f'Stroma IoU: \033[1;35m{tissue_iou[1]:.4f}\033[0m')
            print(f'Lymphocytic infiltrate IoU: \033[1;35m{tissue_iou[2]:.4f}\033[0m')
            print(f'Necrosis IoU: \033[1;35m{tissue_iou[2]:.4f}\033[0m')
            print(f'mIoU: \033[1;35m{miou:.4f}\033[0m')
            print(f'fwIoU: \033[1;35m{fwiou:.4f}\033[0m')
        elif self.args.dataset == 'luad':
            self.pred_big_mask_dict_ms_val = None
            self.cnt_big_mask_dict_ms_val = None
            print(f'Tumor IoU: \033[1;35m{tissue_iou[0]:.4f}\033[0m')
            print(f'Necrosis IoU: \033[1;35m{tissue_iou[1]:.4f}\033[0m')
            print(f'Lymphocytic infiltrate IoU: \033[1;35m{tissue_iou[2]:.4f}\033[0m')
            print(f'Stroma IoU: \033[1;35m{tissue_iou[2]:.4f}\033[0m')
            print(f'mIoU: \033[1;35m{miou:.4f}\033[0m')
            print(f'fwIoU: \033[1;35m{fwiou:.4f}\033[0m')

    def inference_batch(self, batch, metric, pred_big_mask_dict_ms, cnt_big_mask_dict_ms, data_path):
        image_batch, mask_batch, label_batch, name_batch, original_h_batch, original_w_batch = batch['image'], batch['mask'], batch['label'], batch['name'], batch['h'], batch['w']
        result_dict = self(data=image_batch, mask=mask_batch)
        output = result_dict['logits']
        b, c, h, w = output.shape
        
        patch_cls = label_batch.unsqueeze(-1).unsqueeze(-1).expand(b, c, h, w) # [B, C]
        output[patch_cls == 0] = -65504 # min float16
        
        metric(output, mask_batch)

        if self.args.dataset == 'wsss4luad':
            # ----> Process each sample
            for j in range(image_batch.shape[0]):
                original_w, original_h = original_w_batch[j], original_h_batch[j]

                output_ = output[j][:, :original_h, :original_w] # logits, C x 256 x 256
                probs = torch.softmax(output_, dim=0).cpu().numpy()
                probs = probs.transpose(1, 2, 0) # [H, W, C]

                name = name_batch[j]
                image_idx = name.split('_')[0]
                scale = float(name.split('_')[1])
                position = (int(name.split('_')[2]), int(name.split('_')[3].split('-')[0]))

                dict_key = f'{image_idx}_{scale}'

                if dict_key not in pred_big_mask_dict_ms:
                    w, h = Image.open(os.path.join('/'.join(data_path.split('/')[:-1]), 'img', image_idx + '.png')).size
                    w_ = int(w * scale)
                    h_ = int(h * scale)
                    pred_big_mask_dict_ms[dict_key] = np.zeros((h_, w_, 3))
                    cnt_big_mask_dict_ms[dict_key] = np.zeros((h_, w_, 1))
                    
                pred_big_mask_dict_ms[dict_key][position[0]:position[0]+output_.shape[1], position[1]:position[1]+output_.shape[2], :] += probs
                cnt_big_mask_dict_ms[dict_key][position[0]:position[0]+output_.shape[1], position[1]:position[1]+output_.shape[2], :] += 1

    def validation_step(self, batch, batch_idx):
        self.inference_batch(batch, self.valid_iou, self.pred_big_mask_dict_ms_val, self.cnt_big_mask_dict_ms_val, os.path.dirname(self.trainer.val_dataloaders[0].dataset.image_dir))

    def merge_result(self, pred_big_mask_dict_ms, cnt_big_mask_dict_ms, pred_big_mask_dict, cnt_big_mask_dict, data_path):
        # ----> All validation patches are predicted, now we can calculate the final miou
        for k, mask in pred_big_mask_dict_ms.items():
            mask /= cnt_big_mask_dict_ms[k] # [H, W, 3]
            image_idx = k.split('_')[0]

            if image_idx not in pred_big_mask_dict:
                w, h = Image.open(os.path.join('/'.join(data_path.split('/')[:-1]), 'img', image_idx + '.png')).size
                pred_big_mask_dict[image_idx] = np.zeros((h, w, 3))
                cnt_big_mask_dict[image_idx] = np.zeros((h, w, 1))

            mask = F.interpolate(torch.from_numpy(mask.transpose(2, 0, 1)).unsqueeze(0), (h, w), mode='bilinear')[0].numpy().transpose(1, 2, 0)
            pred_big_mask_dict[image_idx][:, :, :] += mask
            cnt_big_mask_dict[image_idx][:, :, :] += 1

        big_mask_iou = mIoUMask()
        for idx, (k, mask_pred) in enumerate(pred_big_mask_dict.items()):
            mask_pred /= cnt_big_mask_dict[k] # hwc
            mask = Image.open(os.path.join('/'.join(data_path.split('/')[:-1]), 'mask', k + '.png'))

            big_mask_iou(torch.from_numpy(mask_pred.transpose(2, 0, 1)).unsqueeze(0), torch.from_numpy(np.array(mask)).unsqueeze(0), probs=True)
        return big_mask_iou

    def validation_epoch_end(self, validation_step_outputs):
        if self.args.dataset == 'wsss4luad':
            val_big_mask_iou = self.merge_result(self.pred_big_mask_dict_ms_val, self.cnt_big_mask_dict_ms_val, self.pred_big_mask_dict_val, self.cnt_big_mask_dict_val, os.path.dirname(self.trainer.val_dataloaders[0].dataset.image_dir))

            # ----> Print and Log val metrics
            tissue_iou = self.valid_iou.Tissue_Intersection_over_Union()
            print('\n' + '-' * 50)
            print("\nExperiment Settings")
            print(f"Dataset: \033[1;34m{self.args.train_image_dirs}\033[0m")
            print(f"Log Path: \033[1;34m{self.args.log_path}\033[0m")

            print('\n' + '-' * 50)
            print(f"\nValidation Result (Patch)")
            print(f'Tumor IoU: \033[1;35m{tissue_iou[0]:.4f}\033[0m')
            print(f'Stroma IoU: \033[1;35m{tissue_iou[1]:.4f}\033[0m')
            print(f'Normal IoU: \033[1;35m{tissue_iou[2]:.4f}\033[0m')
            print(f'mIoU: \033[1;35m{self.valid_iou.Mean_Intersection_over_Union():.4f}\033[0m')
            print(f'fwIoU: \033[1;35m{self.valid_iou.Frequency_Weighted_Intersection_over_Union():.4f}\033[0m')
            print('\n' + '-' * 50)
            self.log(f'validation_tiou_patch_epoch', tissue_iou[0], prog_bar=False)
            self.log(f'validation_siou_patch_epoch', tissue_iou[1], prog_bar=False)
            self.log(f'validation_niou_patch_epoch', tissue_iou[2], prog_bar=False)
            self.log(f'validation_miou_patch_epoch', self.valid_iou.Mean_Intersection_over_Union(), prog_bar=True)
            self.log(f'validation_fwiou_patch_epoch', self.valid_iou.Frequency_Weighted_Intersection_over_Union(), prog_bar=True)
            self.valid_iou.reset()

            val_big_tissue_iou = val_big_mask_iou.Tissue_Intersection_over_Union()
            print('\n' + '-' * 50)
            print(f"\nValidation Result (Big Mask)")
            print(f'Tumor IoU: \033[1;35m{val_big_tissue_iou[0]:.4f}\033[0m')
            print(f'Stroma IoU: \033[1;35m{val_big_tissue_iou[1]:.4f}\033[0m')
            print(f'Normal IoU: \033[1;35m{val_big_tissue_iou[2]:.4f}\033[0m')
            print(f'mIoU: \033[1;35m{val_big_mask_iou.Mean_Intersection_over_Union():.4f}\033[0m')
            print(f'fwIoU: \033[1;35m{val_big_mask_iou.Frequency_Weighted_Intersection_over_Union():.4f}\033[0m')
            print('\n' + '-' * 50)

            self.log(f'validation_tiou_mask_epoch', val_big_tissue_iou[0], prog_bar=False)
            self.log(f'validation_siou_mask_epoch', val_big_tissue_iou[1], prog_bar=False)
            self.log(f'validation_niou_mask_epoch', val_big_tissue_iou[2], prog_bar=False)
            self.log(f'validation_miou_mask_epoch', val_big_mask_iou.Mean_Intersection_over_Union(), prog_bar=True)
            self.log(f'validation_fwiou_mask_epoch', val_big_mask_iou.Frequency_Weighted_Intersection_over_Union(), prog_bar=True)
            val_big_mask_iou.reset()
        elif self.args.dataset == 'bcss':
            val_tissue_iou = self.valid_iou.Tissue_Intersection_over_Union()
            print('\n' + '-' * 50)
            print("\nExperiment Settings")
            print(f"Dataset: \033[1;34m{self.args.train_image_dirs}\033[0m")
            print(f"Log Path: \033[1;34m{self.args.log_path}\033[0m")
            print('\n' + '-' * 50)
            print(f"\nValidation Result (Mask)")
            print(f'Tumor IoU: \033[1;35m{val_tissue_iou[0]:.4f}\033[0m')
            print(f'Stroma IoU: \033[1;35m{val_tissue_iou[1]:.4f}\033[0m')
            print(f'Lymphocytic infiltrate IoU: \033[1;35m{val_tissue_iou[2]:.4f}\033[0m')
            print(f'Necrosis IoU: \033[1;35m{val_tissue_iou[3]:.4f}\033[0m')
            print(f'mIoU: \033[1;35m{self.valid_iou.Mean_Intersection_over_Union():.4f}\033[0m')
            print(f'fwIoU: \033[1;35m{self.valid_iou.Frequency_Weighted_Intersection_over_Union():.4f}\033[0m')
            print('\n' + '-' * 50)
            self.log(f'validation_tmr_mask_epoch', val_tissue_iou[0], prog_bar=False)
            self.log(f'validation_str_mask_epoch', val_tissue_iou[1], prog_bar=False)
            self.log(f'validation_lym_mask_epoch', val_tissue_iou[2], prog_bar=False)
            self.log(f'validation_nec_mask_epoch', val_tissue_iou[3], prog_bar=False)
            self.log(f'validation_miou_mask_epoch', self.valid_iou.Mean_Intersection_over_Union(), prog_bar=True)
            self.log(f'validation_fwiou_mask_epoch', self.valid_iou.Frequency_Weighted_Intersection_over_Union(), prog_bar=True)
            self.valid_iou.reset()
        elif self.args.dataset == 'luad':
            val_tissue_iou = self.valid_iou.Tissue_Intersection_over_Union()
            print('\n' + '-' * 50)
            print("\nExperiment Settings")
            print(f"Dataset: \033[1;34m{self.args.train_image_dirs}\033[0m")
            print(f"Log Path: \033[1;34m{self.args.log_path}\033[0m")

            print('\n' + '-' * 50)
            print(f"\nValidation Result (Mask)")
            print(f'Tumor IoU: \033[1;35m{val_tissue_iou[0]:.4f}\033[0m')
            print(f'Necrosis IoU: \033[1;35m{val_tissue_iou[1]:.4f}\033[0m')
            print(f'Lymphocytic infiltrate IoU: \033[1;35m{val_tissue_iou[2]:.4f}\033[0m')
            print(f'Stroma IoU: \033[1;35m{val_tissue_iou[3]:.4f}\033[0m')
            print(f'mIoU: \033[1;35m{self.valid_iou.Mean_Intersection_over_Union():.4f}\033[0m')
            print(f'fwIoU: \033[1;35m{self.valid_iou.Frequency_Weighted_Intersection_over_Union():.4f}\033[0m')
            print('\n' + '-' * 50)

            self.log(f'validation_tmr_mask_epoch', val_tissue_iou[0], prog_bar=False)
            self.log(f'validation_nec_mask_epoch', val_tissue_iou[1], prog_bar=False)
            self.log(f'validation_lym_mask_epoch', val_tissue_iou[2], prog_bar=False)
            self.log(f'validation_tas_mask_epoch', val_tissue_iou[3], prog_bar=False)
            self.log(f'validation_miou_mask_epoch', self.valid_iou.Mean_Intersection_over_Union(), prog_bar=True)
            self.log(f'validation_fwiou_mask_epoch', self.valid_iou.Frequency_Weighted_Intersection_over_Union(), prog_bar=True)

            self.valid_iou.reset()

    #----> remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items