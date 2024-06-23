GPU=6,
NUM_SAMPLES_1=7200
NUM_SAMPLES_2=7200

for k in 2
do

LOG_DIR=logs/bcss7k2_7k2_run$k


python mosaic_train_v2.py --model DeepLabV3Plus --encoder efficientnet-b6 --lr 0.0001 --gpus $GPU --epochs 25 --batch-size 32 --semi_image_dir data/BCSS-WSSS/training --train_image_dirs data/BCSS-WSSS/mosaic_2_112_run$k/disc_img_r18_e5 data/BCSS-WSSS/bezier224_5_0.2_0.05_1d1_run$k/disc_img_r18_e5 --train_mask_dirs data/BCSS-WSSS/mosaic_2_112_run$k/disc_mask_r18_e5 data/BCSS-WSSS/bezier224_5_0.2_0.05_1d1_run$k/disc_mask_r18_e5 --num_samples $NUM_SAMPLES_1 $NUM_SAMPLES_2 --patch-size 224 --val_data data/BCSS-WSSS/val --test_data data/BCSS-WSSS/test --num-classes 4 --dataset bcss --log_dir $LOG_DIR

CUDA_VISIBLE_DEVICES=$GPU python segmentation_test_v2.py -ckpt $LOG_DIR --gpus=0, --patch-size=224 --test-data data/BCSS-WSSS/test --dataset bcss

CUDA_VISIBLE_DEVICES=$GPU python infer_pseudo_masks_v2.py --checkpoint $LOG_DIR --train-data data/BCSS-WSSS/training --save-dir $LOG_DIR --gpus 0 --batch-size 64 --dataset bcss

python mosaic_train_v2.py --model DeepLabV3Plus --encoder efficientnet-b6 --lr 0.0001 --gpus $GPU --epochs 25 --batch-size 32 --train_image_dirs data/BCSS-WSSS/training data/BCSS-WSSS/mosaic_2_112_run$k/disc_img_r18_e5 data/BCSS-WSSS/bezier224_5_0.2_0.05_1d1_run$k/disc_img_r18_e5 --train_mask_dirs $LOG_DIR/mask data/BCSS-WSSS/mosaic_2_112_run$k/disc_mask_r18_e5 data/BCSS-WSSS/bezier224_5_0.2_0.05_1d1_run$k/disc_mask_r18_e5 --num_samples -1 $NUM_SAMPLES_1 $NUM_SAMPLES_2 --patch-size 224 --val_data data/BCSS-WSSS/val --test_data data/BCSS-WSSS/test --num-classes 4 --dataset bcss --log_dir $LOG_DIR/stage2

CUDA_VISIBLE_DEVICES=$GPU python segmentation_test_v2.py -ckpt $LOG_DIR/stage2 --gpus=0, --patch-size=224 --test-data data/BCSS-WSSS/test --dataset bcss

done