#!/bin/bash

## script adapted and uploaded to FW by krj

##
### Script for Executing rPOP
### More information about this pipeline can be found here: https://github.com/LeoIacca/rPOP
### Script pre-processes PET data (without T1w image) for centilloid calculations
##  

# Load config or inputs manually
CmdName=$(basename "$0")
Syntax="${CmdName} [-c config][-a PETdata][-n][-o Origin][-t Template][-v]"
function sys {
        [ -n "${opt_n}${opt_v}" ] && echo "$@" 1>&2
        [ -n "$opt_n" ] || "$@"
} 
while getopts a:c:i:m:ntv arg
do
        case "$arg" in 
                a|c|n|v)
                        eval "opt_${arg}='${OPTARG:=1}'"
                        ;;
        esac
done
shift $(( $OPTIND - 1))  

# Check if there is a config
# If so, load info from config,
# If not, load data manually
if [ -n "$opt_c" ]
then
        ConfigJsonFile="$opt_c"
else
        ConfigJsonFile="${FLYWHEEL:=.}/config.json"
fi

if [ -n "$opt_a" ]; then
        petdata="$opt_a"   
else
        petdata=$( jq '.inputs.petdata.location.path' "$ConfigJsonFile" | tr -d '"' )
fi

if [ -n "$opt_o" ]; then
        Origin="$opt_o"
else
        Origin=$( jq '.config.origin' "$ConfigJsonFile" | tr -d '"' )
fi

if [ -n "$opt_t" ]; then
        Template="$opt_t"
else
        Template=$( jq '.config.template' "$ConfigJsonFile" | tr -d '"' )
fi

### Data Preprocessing
# Set up data paths
flywheel='/flywheel/v0'
[ -e "$flywheel" ] || mkdir "$flywheel"
rpop_dir='/flywheel/v0/rPOP-master'
[ -e "$rpop_dir" ] || mkdir "$rpop_dir"
data_dir='/flywheel/v0/input'
[ -e "$data_dir" ] || mkdir "$data_dir"
out_dir='/flywheel/v0/output'
[ -e "$out_dir" ] || mkdir "$out_dir"
work_dir='/flywheel/v0/work'
[ -e "$work_dir" ] || mkdir "$work_dir" 
exe_dir='/flywheel/v0/workflows'
[ -e "$exe_dir" ] || mkdir "$exe_dir"

# Now we need to clean and pass the data to the MATLAB function
if [ "$Origin" == "Keep" ]
then
	oropt=1
else
	oropt=2
fi

if [ "$Template" == "All" ]
then
	tpopt=1
elif [ "$Template" == "Florbetapir" ]
then
	tpopt=2
elif [ "$Template" == "Florbetaben" ]
then
	tpopt=3
elif [ "$Template" == "Flutemetamol" ]
then
	tpopt=4
fi

# Create binary masks of both the input and reference data
${FREESURFER_HOME}/bin/mri_synthstrip -i ${petdata} -m ${work_dir}/pet_mask.nii.gz

${FREESURFER_HOME}/bin/mri_synthstrip -i ${rpop_dir}/templates/Template_FBB_all.nii -m ${work_dir}/temp_mask.nii.gz

# Run SPM12 to make sure it's ready for the script
python3 ${exe_dir}/rPOP.py -pet ${petdata} -work ${work_dir} -out ${out_dir} -rpop ${rpop_dir} -tpopt ${tpopt} -origin ${oropt} -exe ${exe_dir}
