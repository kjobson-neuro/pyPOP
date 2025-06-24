#!/usr/bin/env python3

import os
import numpy as np
import nibabel as nb
from scipy import ndimage
from scipy.optimize import minimize
from datetime import datetime
import nipype
import nipype.interfaces.afni as afni
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import subprocess
import argparse
import nilearn
from nilearn import maskers
from nilearn import image
from picsl_greedy import Greedy3D
import SimpleITK as sitk

env = os.environ.copy()
g = Greedy3D()

# Import data
parser = argparse.ArgumentParser(description='Calculate centilloid with only PET data.')

# Set up parser for the PET data and its output
parser.add_argument('-pet', type=str, help="The path to the PET scan.")
parser.add_argument('-work', type=str, help="The path to the work directory.")
parser.add_argument('-rpop', type=str, help="The path to the rPOP master directory.")
parser.add_argument('-origin', type=str, help="Reset origin?")
parser.add_argument('-tpopt', type=int, help="Which template?")
parser.add_argument('-out', type=str, help="The path to the output directory.")
parser.add_argument('-exe', type=str, help="The path to the directory with executable scripts.")
args = parser.parse_args()

# Load the input options
input_file = args.pet
work_dir = args.work
rpop_dir = args.rpop
tpopt = args.tpopt
origin = args.origin
output_dir = args.out
exe_dir = args.exe
temp_dir = os.path.join(rpop_dir, 'templates')

# Define function to load templates 
def load_images(paths):
    images = []
    for path in paths:
        img = nb.load(path)
        images.append(img.get_fdata())
    return images

# Define function to set origin if set by user
def reset_image_origin(file_path):
    # Load NIfTI image
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    
    # Get image dimensions (assuming 3D)
    dim_x, dim_y, dim_z = header.get_data_shape()[:3]
    
    # Calculate center in 0-based voxel coordinates (MATLAB uses 1-based)
    center_voxel = np.array([(dim_x-1)/2, (dim_y-1)/2, (dim_z-1)/2, 1])
    
    # Calculate world coordinates of center using original affine
    center_world = affine @ center_voxel
    
    # Create new affine matrix with updated origin
    new_affine = affine.copy()
    new_affine[:3, 3] = center_world[:3] - (affine[:3, :3] @ center_voxel[:3])
    # Save new image
    new_img = nb.Nifti1Image(data, new_affine, header=img.header)
    new_file_name = os.path.splitext(file_path)[0] + 'img_centered.nii.gz'
    new_file_path = os.path.join(work_dir, new_file_name)
    nb.save(new_img, new_file_path)

# Define cost function to minimize difference between PET template and scan
def cost_function(coefficients, templates, source_image):
    # Ensure source_image is a numpy array
    if isinstance(source_image, nb.Nifti1Image):
        source_image = source_image.get_fdata()
    combined_image = np.sum([coeff * template for coeff, template in zip(coefficients, templates)], axis=0)
    mse = np.mean((combined_image - source_image) ** 2)
    return mse

# Find best linear combination of templates
def find_best_combination(templates, source_image):
    initial_guess = np.array([1.0 / len(templates)] * len(templates))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    result = minimize(cost_function, initial_guess, args=(templates, source_image), method='SLSQP', constraints=constraints)
    return result.x

# Define the main function
def rPOP(input_file, output_dir, set_origin, template, work_dir, temp_dir):
    print("\n\n********** Welcome to pyPOP v1.0 (January 2025) **********")
    print("pyPOP is dependent on:")
    print("*1. AFNI Neuroimaging Suite (https://afni.nimh.nih.gov/)")
    print("*2. Python (https://www.python.org/)")
    print("*3. Greedy: Fast Deformable Registration for 2D and 3D Medical Images (https://greedy.readthedocs.io/en/latest/index.html)")
    print("*** pyPOP is only distributed for academic/research purposes, with NO WARRANTY. ***")
    print("*** pyPOP is not intended for any clinical or diagnostic purposes. ***")

    # Load the templates
    warptempl_all = [
       os.path.join(temp_dir, 'Template_FBP_all.nii'),
       os.path.join(temp_dir, 'Template_FBP_pos.nii'),
       os.path.join(temp_dir, 'Template_FBP_neg.nii'),
       os.path.join(temp_dir, 'Template_FBB_all.nii'),
       os.path.join(temp_dir, 'Template_FBB_pos.nii'),
       os.path.join(temp_dir, 'Template_FBB_neg.nii'),
       os.path.join(temp_dir, 'Template_FLUTE_all.nii'),
       os.path.join(temp_dir, 'Template_FLUTE_pos.nii'),
       os.path.join(temp_dir, 'Template_FLUTE_neg.nii')
    ]

    warptempl_fbp = [
       os.path.join(temp_dir, 'Template_FBP_all.nii'),
       os.path.join(temp_dir, 'Template_FBP_pos.nii'),
       os.path.join(temp_dir, 'Template_FBP_neg.nii')
    ]

    warptemple_fbb = [
       os.path.join(temp_dir, 'Template_FBB_all.nii'),
       os.path.join(temp_dir, 'Template_FBB_pos.nii'),
       os.path.join(temp_dir, 'Template_FBB_neg.nii')
    ]

    warptempl_flute = [
       os.path.join(temp_dir, 'Template_FLUTE_all.nii'),
       os.path.join(temp_dir, 'Template_FLUTE_pos.nii'),
       os.path.join(temp_dir, 'Template_FLUTE_neg.nii')
    ]

    # Reset origin to center of image if it's set
    if set_origin == "Reset":
        reser_image_origin(input_file)
    elif set_origin == "Keep":
        data = nb.load(input_file)
        nb.save(data, f'{work_dir}/img_centered.nii.gz')

    # Template choice
    if tpopt == 1:
        warptempl = warptempl_all
    elif tpopt == 2:
        warptempl = warptempl_fbp
    elif tpopt == 3:
        warptempl = warptemple_fbb
    elif tpopt == 4:
        warptempl = warptempl_flute

    templates = load_images(warptempl)

    pet_data = os.path.join(work_dir, 'img_centered.nii.gz')

    # Create brain masks
    pet_mask = nilearn.masking.compute_brain_mask(pet_data, threshold=0.0, connected=False, opening=2, memory=None, verbose=0, mask_type='whole-brain')
    nb.save(pet_mask, f'{work_dir}/pet_mask.nii.gz')
    temp_mask = nilearn.masking.compute_brain_mask(f'{work_dir}/composite_template.nii.gz', threshold=0.0, connected=False, opening=2, memory=None, verbose=0, mask_type='whole-brain')
    nb.save(temp_mask, f'{work_dir}/temp_mask.nii.gz')

    # Perform affine registration
    temp_reg = os.path.join(temp_dir, 'Template_FBB_all.nii')
    
    prefix = 'init_reg'

    img_fixed = sitk.ReadImage(temp_reg)
    img_moving = sitk.ReadImage(pet_data)
    fixed_mask = sitk.ReadImage(temp_mask)
    moving_mask= sitk.ReadImage(pet_mask)

    warpedimg = f'{work_dir}/{prefix}.nii.gz'

    g.execute('-i my_fixed my_moving '
           '-ia-image-centers '
           '-gm fmask '
           '-mm mmask '
           '-a -dof 6 -n 100x40x10 -m MI '
           '-o affine',
           my_fixed = img_fixed, my_moving = img_moving, fmask = fixed_mask, mmask = moving_mask,
           affine = None)

    g.execute('-rf my_fixed '
          '-rm my_moving warpedimg '
          '-r affine')

    # Find best linear combination
    coefficients = find_best_combination(templates, warpedimg)

    # Create composite template
    composite_template = np.sum([coeff * template for coeff, template in zip(coefficients, templates)], axis=0)

    # Save composite template
    # Use the affine from one of the templates (e.g., the first one)
    template_affine = nb.load(warptempl[0]).affine
    composite_img = nb.Nifti1Image(composite_template, template_affine)
    nb.save(composite_img, f'{work_dir}/composite_template.nii.gz')

    # Warp the image to MNI space using ANTs
    full_prefix = 'w_pet'
    deformedimg = f'{work_dir}/{full_prefix}.nii.gz'
    # Perform registration
    g.execute('-i my_fixed my_moving '
          '-it affine -dof 12 -n 100x40x10 -m MI -s 20.0vox 10.0vox '
          '-gm fmask '
          '-mm mmask '
          '-o deform',
           my_fixed = img_fixed, my_moving = img_moving, fmask = fixed_mask, mmask = moving_mask,
           my_warp = None)
    # Reslice
    g.execute('-rf my_fixed -rm my_moving deformedimg '
          '-r deform affine',
          my_resliced = None)

    # Estimate FWHM using AFNI's 3dFWHMx
    afni_out = 'sw_pet_afni'
    subprocess.run([f"{exe_dir}/afni.sh",
                f"{work_dir}", f"{afni_out}"])

    fwhm_file = f'{work_dir}/sw_pet_afni_automask.txt'
    # Read FWHM estimations
    fwhm_data = np.loadtxt(fwhm_file)

    # Extract only the first row for old FWHM calc
    fwhm_x, fwhm_y, fwhm_z = fwhm_data[0, 0:3]

    # Calculate smoothing filters
    def calc_filter(fwhm):
        return np.sqrt(max(0, 10**2 - fwhm**2)) if fwhm < 10 else 0

    filter_x = calc_filter(fwhm_x)
    filter_y = calc_filter(fwhm_y)
    filter_z = calc_filter(fwhm_z)

    # Apply smoothing
    sigma = (filter_x / 2.355, filter_y / 2.355, filter_z / 2.355)

    smoothed_prefix = 'sw_pet'
    in_img = os.path.join(work_dir, f'{full_prefix}.nii.gz')
    smoothed_img = nilearn.image.smooth_img(in_img,fwhm=[filter_x, filter_y, filter_z])  # Direct FWHM input in mm

    nb.save(smoothed_img, f'{output_dir}/{smoothed_prefix}.nii.gz')

    # Calculate wtx
    ctx = os.path.join(rpop_dir, 'Centiloid_Std_VOI/nifti/2mm', 'voi_ctx_2mm.nii')
    ctx_img = nb.load(ctx)

    wc = os.path.join(rpop_dir, 'Centiloid_Std_VOI/nifti/2mm', 'voi_WhlCbl_2mm.nii')  
    wc_img = nb.load(wc)


    # Save results to CSV
    results = {
        'SubjectID': ['sw_pet.nii.gz'],
        'avg_ctx_voi_bin': [''],
        'avg_wc_voi_bin': [''],
        'NeocorticalSUVR': [''],
        'Centilloid': [''],
        'EstimatedFWHMx': [fwhm_x],
        'EstimatedFWHMy': [fwhm_y],
        'EstimatedFWHMz': [fwhm_z],
        'FWHMfilterappliedx': [filter_x],
        'FWHMfilterappliedy': [filter_y],
        'FWHMfilterappliedz': [filter_z],
        'AFNIEstimationRerunMod': ['0']
    }

    import pandas as pd
    print(results)
    df = pd.DataFrame(results)
    csv_file = os.path.join(output_dir, f'PYrPOP_{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}.csv')
    df.to_csv(csv_file, index=False)

    print("\nPYrPOP just finished! Warped and differentially smoothed AC PET images were generated.")
    print("Lookup the .csv database to assess FWHM estimations and filters applied.\n")

# Execute:
rPOP(input_file, output_dir, origin, tpopt, work_dir, temp_dir)

