
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

logging.basicConfig(level=logging.INFO)



# %%
snap_dir = "/mnt/efs/shared_data/hack/lsd/lsd_exp1/snapshot"

f = zarr.open(os.path.join(snap_dir, 'newbatch_2001.zarr'), 'r')


#%%

print("Shape of 'gt':", f['gt'].shape)
print("Shape of 'gt_affinities':", f['gt_affinities'].shape)
print("Shape of 'pred_affinities':", f['pred_affinities'].shape)


# %%

# Plotting the GT and affinitites

# Select the first element from the batch and the first slice in the z-dimension
batch_idx = 0
z_idx = 0

fig, axes = plt.subplots(3, 4, figsize=(25, 18), sharex=True, sharey=True)

# Plot gt four times in the top row
for ax in axes[0]:
    ax.imshow(f['gt'][batch_idx, z_idx], cmap='viridis')
    ax.set_title('gt')

# Plot each channel of gt_affinities
for i, ax in enumerate(axes[1]):
    if i == 0:
        ax.imshow(np.sum(f['gt_affinities'][batch_idx, :, z_idx], axis=0), cmap='viridis')
        ax.set_title('gt_affinities summed')
    else:
        ax.imshow(f['gt_affinities'][batch_idx, i-1, z_idx], cmap='viridis')
        ax.set_title(f'gt_affinities channel {i-1}')

# Plot each channel of pred_affinities
for i, ax in enumerate(axes[2]):
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


#%%

pred_affs = f['pred_affinities'][0]

# %%

pred_affs = f['pred_affinities'][0]

ws_affs = np.stack([
    np.zeros_like(pred_affs[0]),
    pred_affs[0],
    pred_affs[1]]
)

threshold = 0.5
segmentation = get_segmentation(ws_affs, threshold)


# %%

z_slice = 0  # example Z slice for visualization

fig, axes = plt.subplots(4, 4, figsize=(20, 20), sharex=True, sharey=True)


# 1st row: GT displayed 4 times for layout uniformity
for ax in axes[0]:
    ax.imshow(f['gt'][0, z_slice], cmap='viridis')
    ax.set_title("GT")

# 2nd row: GT affinities (3 channels)
axes[1][0].imshow(f['gt_affinities'][0, 0, z_slice], cmap='viridis')
axes[1][0].set_title("GT Affinity 1")
axes[1][1].imshow(f['gt_affinities'][0, 1, z_slice], cmap='viridis')
axes[1][1].set_title("GT Affinity 2")
axes[1][2].imshow(f['gt_affinities'][0, 2, z_slice], cmap='viridis')
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
