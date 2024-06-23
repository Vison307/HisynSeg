#!/bin/bash

for run in 2
do
for idx in 0 1 2 3
do

CUDA_VISIBLE_DEVICES=4, python create_dataset_mosaic_disc_bcss.py --idx $idx --run $run

done
done