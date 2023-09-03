
#%%
import gunpowder as gp
import h5py
import io
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import requests
import torch
import waterz
import zarr

from funlib.learn.torch.models import UNet, ConvPass
from gunpowder.torch import Predict
from scipy.ndimage import label, measurements
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed
from scipy.ndimage import zoom
from skimage.transform import resize
logging.basicConfig(level=logging.INFO)



# %%
snap_dir = "/mnt/efs/shared_data/hack/lsd/aff_exp2/snapshot"

f = zarr.open(os.path.join(snap_dir, 'newbatch_951.zarr'), 'r')


#%%

print("Shape of 'raw':", f['raw'].shape)
print("Shape of 'gt':", f['gt'].shape)
print("Shape of 'fg':", f['fg_mask'].shape)
print("Shape of 'gt_affinities':", f['gt_affinities'].shape)
print("Shape of 'pred_affinities':", f['pred_affinities'].shape)


# %%


def resize_to_match(target_shape, raw_image):
    # Calculate the zoom factors for each axis
    factors = [t/r for t, r in zip(target_shape, raw_image.shape)]
    
    # Resize using scipy's zoom function
    return zoom(raw_image, factors, order=1)  # order=1 for bilinear interpolation

# Assuming you have raw and gt loaded
raw_image = f['raw'][1, 0]  # Select one batch and remove channel dim
target_shape = (8, 88, 88)

resized_raw = resize_to_match(target_shape, raw_image)

#%%

# Plotting the raw, GT, affinities, and fg_mask

# Select the first element from the batch
batch_idx = 2
z_idx = 5  # example z-slice for 'gt', 'fg_mask', 'gt_affinities', and 'pred_affinities'
raw_z_idx = z_idx * 3  # adjusted z-slice for 'raw'

fig, axes = plt.subplots(5, 4, figsize=(25, 20), sharex=True, sharey=True)

# Plot raw in the first row
axes[0][0].imshow(f['raw'][batch_idx, 0, raw_z_idx], cmap='viridis')
axes[0][0].set_title('raw')
# Empty plots for rest of the columns in first row
for ax in axes[0][1:]:
    ax.axis('off')

# Plot gt in the second row
axes[1][0].imshow(f['gt'][batch_idx, z_idx], cmap='viridis')
axes[1][0].set_title('gt')
# Empty plots for rest of the columns in second row
for ax in axes[1][1:]:
    ax.axis('off')

# Plot fg_mask in the third row
axes[2][0].imshow(f['fg_mask'][batch_idx, z_idx], cmap='viridis')
axes[2][0].set_title('fg_mask')
# Empty plots for rest of the columns in third row
for ax in axes[2][1:]:
    ax.axis('off')

# Plot each channel of gt_affinities in the fourth row
for i, ax in enumerate(axes[3]):
    if i == 0:
        ax.imshow(np.sum(f['gt_affinities'][batch_idx, :, z_idx], axis=0), cmap='viridis')
        ax.set_title('gt_affinities summed')
    else:
        ax.imshow(f['gt_affinities'][batch_idx, i-1, z_idx], cmap='viridis')
        ax.set_title(f'gt_affinities channel {i-1}')

# Plot each channel of pred_affinities in the fifth row
for i, ax in enumerate(axes[4]):
    if i == 0:
        ax.imshow(np.sum(f['pred_affinities'][batch_idx, :, z_idx], axis=0), cmap='viridis')
        ax.set_title('pred_affinities summed')
    else:
        ax.imshow(f['pred_affinities'][batch_idx, i-1, z_idx], cmap='viridis')
        ax.set_title(f'pred_affinities channel {i-1}')

plt.tight_layout()
plt.show()

#%%

def watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        id_offset=0,
        min_seed_distance=10):

    max_filtered = maximum_filter(boundary_distances, size=min_seed_distance)
    maxima = (max_filtered == boundary_distances)
    seeds, n = label(maxima)

    print(f"Found {n} fragments")

    if n == 0:
        return np.zeros_like(boundary_distances, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    fragments = watershed(
        boundary_distances.max() - boundary_distances,
        markers=seeds,
        mask=boundary_mask)

    return fragments.astype(np.uint64), n + id_offset

def watershed_from_affinities(
        affs,
        max_affinity_value=1.0,
        id_offset=0,
        min_seed_distance=10):

    mean_affs = 0.5 * (affs[1] + affs[2])
    boundary_mask = mean_affs > 0.5 * max_affinity_value
    boundary_distances = distance_transform_edt(boundary_mask)

    return watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        id_offset=id_offset,
        min_seed_distance=min_seed_distance)


def get_segmentation(affinities, threshold):
    fragments = watershed_from_affinities(affinities)[0]
    thresholds = [threshold]

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),
        fragments=fragments,
        thresholds=thresholds,
    )

    return next(generator)
     


#%%

pred_affs = f['gt_affinities'][4]


print("Shape of 'pred_affinities':", f['pred_affinities'].shape)


# %%

pred_affs = f['pred_affinities'][batch_idx]

ws_affs = np.stack([
    np.zeros_like(pred_affs[0]),
    pred_affs[0],
    pred_affs[1]]
)

threshold = 0.5
segmentation = get_segmentation(ws_affs, threshold)


# %%

    # def resize_3d_batch(batch_data, target_shape):
    # resized_list = []
    # for i in range(batch_data.shape[0]):
    #     resized_img = resize(batch_data[i], target_shape, mode='reflect', anti_aliasing=True)
    #     resized_list.append(resized_img)
    # return np.stack(resized_list, axis=0)

    # target_shape = (1, 8, 88, 88)
    # resized_raw = resize_3d_batch(f['raw'], target_shape)
    # resized_raw_no_channel = resized_raw[:, 0, ...]


def center_crop_3d(data, target_shape):
    z_crop = (data.shape[2] - target_shape[0]) // 2
    y_crop = (data.shape[3] - target_shape[1]) // 2
    x_crop = (data.shape[4] - target_shape[2]) // 2

    return data[:, :, z_crop:z_crop + target_shape[0], y_crop:y_crop + target_shape[1], x_crop:x_crop + target_shape[2]]

cropped_raw = center_crop_3d(f['raw'], (8, 88, 88))

cropped_raw_no_channel = cropped_raw[:, 0, ...]


#%%

print("Shape of 'raw':", f['raw'].shape)
print("Shape of 'gt':", f['gt'].shape)
print("Shape of 'cropped':", cropped_raw.shape)
print("Shape of 'cropped raw_no_channel':", cropped_raw_no_channel.shape)


#%%

z_slice = 1  # example Z slice for visualization

fig, axes = plt.subplots(4, 4, figsize=(20, 20), sharex=True, sharey=True)


# 1st row: First, display the resized_raw, then GT displayed 3 times for layout uniformity
axes[0, 0].imshow(cropped_raw_no_channel[batch_idx, z_slice], cmap='viridis')
axes[0, 0].set_title("Raw")

for ax in axes[0][1:]:
    ax.imshow(f['gt'][batch_idx, z_slice], cmap='viridis')
    ax.set_title("GT")

# 2nd row: GT affinities (3 channels)
axes[1][0].imshow(f['gt_affinities'][batch_idx, 0, z_slice], cmap='viridis')
axes[1][0].set_title("GT Affinity 1")
axes[1][1].imshow(f['gt_affinities'][batch_idx, 1, z_slice], cmap='viridis')
axes[1][1].set_title("GT Affinity 2")
axes[1][2].imshow(f['gt_affinities'][batch_idx, 2, z_slice], cmap='viridis')
axes[1][2].set_title("GT Affinity 3")
axes[1][3].imshow(np.sum(f['gt_affinities'][0, :, z_slice], axis=0), cmap='viridis')
axes[1][3].set_title("GT Affinities Summed")

# 3rd row: Pred affinities (3 channels)
axes[2][0].imshow(pred_affs[0, z_slice], cmap='viridis')
axes[2][0].set_title("Pred Affinity 1")
axes[2][1].imshow(pred_affs[1, z_slice], cmap='viridis')
axes[2][1].set_title("Pred Affinity 2")
axes[2][2].imshow(pred_affs[2, z_slice], cmap='viridis')
axes[2][2].set_title("Pred Affinity 3")
axes[2][3].imshow(np.sum(pred_affs[:, z_slice], axis=0), cmap='viridis')
axes[2][3].set_title("Pred Affinities Summed")

# 4th row: Segmentation
axes[3][0].imshow(segmentation[z_slice], cmap='viridis')
axes[3][0].set_title("Segmentation")
# Rest of the axes in 4th row are empty (or you can fill them with other data if needed)
for ax in axes[3][1:]:
    ax.axis('off')

plt.tight_layout()
plt.show()
# %%
f['raw'].shape
# %%
