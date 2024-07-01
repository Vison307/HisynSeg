#!/bin/bash

for run in {1..4}
do
for idx in 2 3 4 5
do

CUDA_VISIBLE_DEVICES=3, python create_dataset_mosaic_disc_luad.py --idx $idx --run $run

done
done