#!/bin/bash

for run in {1..4}
do
CUDA_VISIBLE_DEVICES=4, python create_bezier_dataset_disc_luad.py --run $run
done