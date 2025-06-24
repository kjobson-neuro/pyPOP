import nibabel as nib
import numpy as np
from scipy.optimize import minimize
import argparse

# Import data
parser = argparse.ArgumentParser(description='Calculate centilloid with only PET data.')

# Set up parser for the PET data and its output
parser.add_argument('-pet', type=str, help="The path to the PET scan.")
parser.add_argument('-work', type=str, help="The path to the work directory.")
parser.add_argument('-templates', type=str, help="The path to the templates.")
parser.add_argument('-origin', type=str, help="Reset origin?")
parser.add_argument('-tpopt', type=int, help="Which template?")
parser.add_argument('-out', type=str, help="The path to the output directory.")

args = parser.parse_args()

# Load the input options
input_file = args.pet
work_dir = args.work
temp_dir = args.templates
tpopt = args.tpopt
origin = args.origin
output_dir = args.out

# Load templates and source image
def load_images(paths):
    images = []
    for path in paths:
        img = nib.load(path)
        images.append(img.get_fdata())
    return images

# Define cost function to minimize
def cost_function(coefficients, templates, source_image):
    # Calculate linear combination of templates
    combined_image = np.sum([coeff * template for coeff, template in zip(coefficients, templates)], axis=0)
    
    # Calculate mean squared error between combined image and source image
    mse = np.mean((combined_image - source_image) ** 2)
    return mse

# Find best linear combination of templates
def find_best_combination(templates, source_image):
    # Initial guess for coefficients (e.g., equal weights)
    initial_guess = np.array([1.0 / len(templates)] * len(templates))
    
    # Constraints: coefficients should sum to 1 (optional)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Minimize cost function
    result = minimize(cost_function, initial_guess, args=(templates, source_image), method='SLSQP', constraints=constraints)
    
    return result.x

# Example usage
template_paths = ['path/to/template1.nii', 'path/to/template2.nii']
source_path = 'path/to/source_image.nii'

templates = load_images(template_paths)
source_image = nib.load(source_path).get_fdata()

coefficients = find_best_combination(templates, source_image)

# Create composite template
composite_template = np.sum([coeff * template for coeff, template in zip(coefficients, templates)], axis=0)

# Apply normalization using composite template
# This step depends on your specific normalization algorithm

