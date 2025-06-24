#!/bin/bash

work_dir=$1
out_name=$2

/opt/afni/install/3dFWHMx -automask -2difMAD -acf -ShowMeClassicFWHM -input ${work_dir}/w_pet.nii.gz -out ${out_name}_subbricks.out > ${work_dir}/${out_name}_automask.txt

