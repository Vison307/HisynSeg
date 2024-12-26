GPU=3,
NUM_SAMPLES=3600

for k in 0
do
LOG_DIR=logs/wsss4luad_run${k}

python mosaic_train_v2.py --model DeepLabV3Plus --encoder efficientnet-b6 --lr 0.0001 --gpus $GPU --epochs 60 --batch-size 32 --semi_image_dir data/WSSS4LUAD/1.training --train_image_dirs data/WSSS4LUAD/mosaic_2_112_run${k}/disc_img_r18_e5 data/WSSS4LUAD/bezier224_5_0.2_0.05_1d1_run${k}/disc_img_r18_e5 --train_mask_dirs data/WSSS4LUAD/mosaic_2_112_run${k}/disc_mask_r18_e5 data/WSSS4LUAD/bezier224_5_0.2_0.05_1d1_run${k}/disc_mask_r18_e5 --num_samples $NUM_SAMPLES $NUM_SAMPLES --patch-size 224 --val_data data/WSSS4LUAD/2.validation/patches_224_224 --test_data data/WSSS4LUAD/3.testing/patches_224_112 --num-classes 3 --dataset wsss4luad --log_dir $LOG_DIR

CUDA_VISIBLE_DEVICES=$GPU python segmentation_test_v2.py -ckpt $LOG_DIR --gpus=0, --patch-size=224 --test-data data/WSSS4LUAD/3.testing/patches_224_112 --dataset wsss4luad

done