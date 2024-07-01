#!/bin/bash

for N in 40
do
for run in 5
do
for idx in 0 1
do
CUDA_VISIBLE_DEVICES=4, python create_dataset_mosaic_disc_luad_limit_N.py --idx $idx --run $run --N_sample $N
done
done
done