#/bin/bash

for run in {0..4}
do
CUDA_VISIBLE_DEVICES=0, python3 segmentation_test_v2.py -ckpt ./logs/wsss4luad_run$run --gpus=0, --patch-size=224 --test-data data/WSSS4LUAD/3.testing/patches_224_112 --dataset wsss4luad
done

for run in {0..4}
do
CUDA_VISIBLE_DEVICES=0, python3 segmentation_test_v2.py -ckpt ./logs/luad_run$run --gpus=0, --patch-size=224 --test-data data/LUAD-HistoSeg/test --dataset luad
done

for run in {0..4}
do
CUDA_VISIBLE_DEVICES=0, python3 segmentation_test_v2.py -ckpt ./logs/bcss_run$run --gpus=0, --patch-size=224 --test-data data/BCSS-WSSS/test --dataset bcss
done

cp -r logs/ /opt/app/outputs/