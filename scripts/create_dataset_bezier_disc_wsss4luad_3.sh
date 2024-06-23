#!/bin/bash

for run in {1..4} # 
do
for idx in 0 1
do
CUDA_VISIBLE_DEVICES=4, python create_bezier_dataset_disc_3.py --idx $idx --run $run
done
done