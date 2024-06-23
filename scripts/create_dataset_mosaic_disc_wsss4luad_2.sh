#!/bin/bash

for run in 1 2 3 4
do
for idx in 0 1
do

CUDA_VISIBLE_DEVICES=7, python create_dataset_mosaic_disc_2.py --idx $idx --run $run

done
done