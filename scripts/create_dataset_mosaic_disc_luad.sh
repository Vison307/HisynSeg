#!/bin/bash

for run in {2..4}
do
for idx in 0 1
do

CUDA_VISIBLE_DEVICES=6, python create_dataset_mosaic_disc_luad.py --idx $idx --run $run

done
done