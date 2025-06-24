#!/bin/bash -l
# Script for running ANTs registration on PET and template data
# Called from rPOP.py, written by krj

fixed="$1"
moving="$2"
transform="$3"
work_dir="$4"
prefix="$5"
pet_mask="$6"
temp_mask="$7"

/Users/katiejobson/Downloads/ants-2.5.4/bin/antsRegistration -d 3 --output "[${work_dir}/${prefix},${work_dir}/${prefix}.nii.gz]" --initial-moving-transform "[${transform}]" -s 1x1x1x1 --use-histogram-matching 0 \
                 -t "BSplineSyN[0.1,20x20x20,0,3]" -x "[$temp_mask,$pet_mask]" -m "CC[$fixed,$moving,1,4]" -c "[100x70x50x20,1e-6,10]" -f 2x2x2x2 -o "[${work_dir}/${prefix},${work_dir}/${prefix}.nii.gz]" -v

