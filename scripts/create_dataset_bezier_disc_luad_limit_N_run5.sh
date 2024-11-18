#!/bin/bash

for N in 40 # 20 30 40
do
for run in 5
do
CUDA_VISIBLE_DEVICES=4, python create_bezier_dataset_disc_luad_limit_N.py --run $run --N_sample $N
done
done 