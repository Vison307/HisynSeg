GPU=5,
NUM_SAMPLES_1=3600
NUM_SAMPLES_2=3600

for r in 3 4 5
do
run=4
LOG_DIR=logs/luad_no_disc3k6_3k6_run${run}_$r

python mosaic_train_v2.py --model DeepLabV3Plus --encoder efficientnet-b6 --lr 0.0001 --gpus $GPU --epochs 25 --batch-size 32 --semi_image_dir data/LUAD-HistoSeg/train --train_image_dirs data/LUAD-HistoSeg/mosaic_2_112_run${run}/img data/LUAD-HistoSeg/bezier224_5_0.2_0.05_1d1_run${run}/img --train_mask_dirs data/LUAD-HistoSeg/mosaic_2_112_run${run}/mask data/LUAD-HistoSeg/bezier224_5_0.2_0.05_1d1_run${run}/mask --num_samples $NUM_SAMPLES_1 $NUM_SAMPLES_2 --patch-size 224 --val_data data/LUAD-HistoSeg/val --test_data data/LUAD-HistoSeg/test --num-classes 4 --dataset luad --log_dir $LOG_DIR

CUDA_VISIBLE_DEVICES=$GPU python segmentation_test_v2.py -ckpt $LOG_DIR --gpus=0, --patch-size=224 --test-data data/LUAD-HistoSeg/test --dataset luad

CUDA_VISIBLE_DEVICES=$GPU python infer_pseudo_masks_v2.py --checkpoint $LOG_DIR --train-data data/LUAD-HistoSeg/train --save-dir $LOG_DIR --gpus 0 --batch-size 64 --dataset luad

python mosaic_train_v2.py --model DeepLabV3Plus --encoder efficientnet-b6 --lr 0.0001 --gpus $GPU --epochs 25 --batch-size 32 --train_image_dirs data/LUAD-HistoSeg/train data/LUAD-HistoSeg/mosaic_2_112_run${run}/img data/LUAD-HistoSeg/bezier224_5_0.2_0.05_1d1_run${run}/img --train_mask_dirs $LOG_DIR/mask data/LUAD-HistoSeg/mosaic_2_112_run${run}/mask data/LUAD-HistoSeg/bezier224_5_0.2_0.05_1d1_run${run}/mask --num_samples -1 $NUM_SAMPLES_1 $NUM_SAMPLES_2 --patch-size 224 --val_data data/LUAD-HistoSeg/val --test_data data/LUAD-HistoSeg/test --num-classes 4 --dataset luad --log_dir $LOG_DIR/stage2

CUDA_VISIBLE_DEVICES=$GPU python segmentation_test_v2.py -ckpt $LOG_DIR/stage2 --gpus=0, --patch-size=224 --test-data data/LUAD-HistoSeg/test --dataset luad

done