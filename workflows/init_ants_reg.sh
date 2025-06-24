#!/bin/bash -l
# Script for running ANTs registration on PET and template data
# Called from rPOP.py, written by krj

fixed="$1"
moving="$2"
work_dir="$3"
prefix="$4"

echo "ANTSPATH in script: $ANTSPATH"

${ANTSPATH}/antsRegistration -d 3 -m "Mattes[${fixed},${moving},1,32,random,0.1]" -t "Rigid[0.2]" -c "[1000x1000x1000,1.e-7,20]" -s 4x2x0 -r "[${fixed},${moving},0]" -f 4x2x1 -a 0 -o "[${work_dir}/${prefix},${work_dir}/${prefix}.nii.gz,${work_dir}/${prefix}_inverse.nii.gz]" -v
