import os
import torch
from torch import nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

class SmpModel(nn.Module):
    def __init__(self, model_name="DeepLabV3Plus", encoder_name="efficientnet-b6", num_classes=3, encoder_depth=None, w1=1.0, w2=1.0, w3=1.0):
        super().__init__()
        self.num_classes = num_classes
        build_params = dict(
            arch=model_name,
            encoder_name=encoder_name,
            in_channels=3,
            classes=num_classes,
        )
        if encoder_depth is not None:
            build_params.update(encoder_depth=encoder_depth)
        if 'Unet' in model_name:
            build_params.update(decoder_attention_type='scse')
    
        model = smp.create_model(**build_params)   
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.segmentation_head = model.segmentation_head 

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=self.encoder.out_channels[-1], out_channels=num_classes, kernel_size=1),
        )
        self.cls_loss = nn.BCEWithLogitsLoss()

        self.train_dice = DiceLoss(mode='multiclass', ignore_index=num_classes)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, **kwargs):
        x = kwargs['data'] # b c h w
        features = self.encoder(x) # [32, 3, 224, 224], [32, 56, 112, 112], [32, 40, 56, 56], [32, 72, 28, 28], [32, 200, 14, 14], [32, 576, 14, 14]

        # for i in range(len(features)):
        #     print(f'features[{i}].shape: {features[i].shape}')

        decoder_output = self.decoder(*features)
        logits = self.segmentation_head(decoder_output) # b, num_cls, h, w
        result_dict = {'logits': logits}
        loss = 0
        
        # classification loss
        cls_features = features[-1] # [32, 576, 14, 14]
        cam = self.classifier(cls_features) # [32, 3, 14, 14]
        cls_logits = F.adaptive_avg_pool2d(cam, output_size=1).squeeze() # [32, 3]
        result_dict.update({'cls_logits': cls_logits, 'cam': cam})

        if 'label' not in kwargs: # inference
            return result_dict
        label = kwargs['label']
        loss_cls = self.w1 * self.cls_loss(cls_logits, label)
        result_dict.update({'loss_cls': loss_cls})

        loss = loss + loss_cls
                  
        # segmentation loss for labeled data
        if 'mask' in kwargs: # fully-supervised
            mask = kwargs['mask'] # b h w
            loss_seg = self.w2 * self.train_dice(logits, mask)
            result_dict.update({'loss_seg': loss_seg})
            loss = loss + loss_seg
        # reg loss for semi
        else:
            logits_interp = F.interpolate(logits, size=(cam.shape[-2], cam.shape[-1]), mode='bilinear', align_corners=False) # [32, 3, 14, 14]
            logits_interp_softmax = torch.softmax(logits_interp, dim=1) # [32, 3, 14, 14]
            cam_softmax = torch.softmax(cam, dim=1) # [32, 3, 14, 14]
            regularization = self.w3 * torch.mean(torch.abs(logits_interp_softmax- cam_softmax))
            result_dict.update({'reg': regularization})       
            loss = loss + regularization

        result_dict.update({'loss': loss})

        return result_dict      
        

if __name__ == '__main__':
    x = torch.randn((32, 3, 224, 224))
    mask = torch.randint(low=0, high=3, size=(32, 224, 224))
    label = torch.randint(low=0, high=2, size=(32, 3)).float()
    model = SmpModel(model_name="DeepLabV3Plus", encoder_name="efficientnet-b6", num_classes=3)

    result_dict = model(data=x, label=label)
    logits = result_dict['logits']
    loss = result_dict.get('loss', torch.Tensor([0]))
    loss_cls = result_dict.get('loss_cls', torch.Tensor([0]))
    loss_seg = result_dict.get('loss_seg', torch.Tensor([0]))
    regularization = result_dict.get('reg', torch.Tensor([0]))

    print(f'logits.shape: {logits.shape}; loss.item(): {loss.item()}; loss_cls.item(): {loss_cls.item()}; loss_seg.item(): {loss_seg.item()}; regularization.item(): {regularization.item()}')