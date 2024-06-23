#!/bin/bash

for run in 2
do
for idx in 0 1
do

CUDA_VISIBLE_DEVICES=7, python create_dataset_mosaic_disc_3.py --idx $idx --run $run

done
done