#!/bin/bash

for run in {1..4}
do
CUDA_VISIBLE_DEVICES=6, python create_bezier_dataset_disc_bcss.py --run $run
done