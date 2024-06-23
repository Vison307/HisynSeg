#!/bin/bash

for run in {5..9} # 
do
for idx in 0 1
do
CUDA_VISIBLE_DEVICES=7, python create_bezier_dataset_disc.py --idx $idx --run $run
done
done