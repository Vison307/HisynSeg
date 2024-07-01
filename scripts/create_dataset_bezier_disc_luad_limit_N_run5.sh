#!/bin/bash

for N in 10 # 20 30 40
do
for run in 4
do
CUDA_VISIBLE_DEVICES=4, python create_bezier_dataset_disc_luad_limit_N.py --run $run --N_sample $N
done
done 