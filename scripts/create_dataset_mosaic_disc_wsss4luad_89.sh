#!/bin/bash

for run in 8 9
do
for idx in 0 1
do

CUDA_VISIBLE_DEVICES=5, python create_dataset_mosaic_disc.py --idx $idx --run $run

done
done