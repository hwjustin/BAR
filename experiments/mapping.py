import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse

def map_flat_to_3d(flat_data, mask_path='mask/nsdgeneral_subj07.nii.gz'):
    nifti = nib.load(mask_path)
    mask = nifti.get_fdata()
    
    original_affine = nifti.affine
    

    print("Affine matrix:\n", original_affine)
  
    mask_binary = mask > 0
    
    print(f"Mask shape: {mask_binary.shape}")
    print(f"Mask data type: {mask_binary.dtype}")
    print(f"Number of non-zero voxels: {np.sum(mask_binary)}")
    print(f"Number of zero voxels: {np.size(mask_binary) - np.sum(mask_binary)}")
    
    
    n_voxels = np.sum(mask_binary)
    print(f"Flat data length: {len(flat_data)}, Number of non-zero voxels in the mask: {n_voxels}")
    assert len(flat_data) == n_voxels, "Flat data length does not match the number of non-zero voxels in the mask."
    
    mapped_data = np.zeros(mask.shape)
    
    mapped_data[mask_binary] = flat_data
    
    return mapped_data, original_affine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--concept", type=str, default="person")
    args = parser.parse_args()

    data = np.load(f'results/bar/feature_importance/subj0{args.subj}/{args.concept}/attributions.npz')
    attributions = data['attributions']

    mean_attributions = np.abs(attributions).mean(axis=0)

    # Apply a power transformation to highlight top attributions
    power_exponent = 3 
    power_attributions = np.power(mean_attributions, power_exponent)

    min_val = np.min(power_attributions)
    max_val = np.max(power_attributions)
    normalized_flat_data = 255 * (power_attributions - min_val) / (max_val - min_val)

    mapped_data, original_affine = map_flat_to_3d(normalized_flat_data, mask_path=f'data/masks/nsdgeneral_subj0{args.subj}.nii.gz')
    
    new_nifti = nib.Nifti1Image(mapped_data, affine=original_affine)
    nib.save(new_nifti, f'results/bar/feature_importance/subj0{args.subj}/{args.concept}/mapped_data.nii.gz')
    print("Mapped data saved as 'mapped_data.nii.gz'")
