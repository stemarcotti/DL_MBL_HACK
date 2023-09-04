# %%
import gunpowder as gp
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
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

f = zarr.open(os.path.join(snap_dir, "newbatch_5051.zarr"), "r")


# %%


def center_crop_3d(data, target_shape):
    z_crop = (data.shape[2] - target_shape[0]) // 2
    y_crop = (data.shape[3] - target_shape[1]) // 2
    x_crop = (data.shape[4] - target_shape[2]) // 2

    return data[
        :,
        :,
        z_crop : z_crop + target_shape[0],
        y_crop : y_crop + target_shape[1],
        x_crop : x_crop + target_shape[2],
    ]




cropped_raw = center_crop_3d(f["raw"], (8, 88, 88))

cropped_raw = cropped_raw[:, 0, ...]


# %%

print("Shape of 'raw':", f["raw"].shape)
print("Shape of 'cropped raw':", cropped_raw.shape)
print("Shape of 'gt':", f["gt"].shape)
print("Shape of 'fg':", f["fg_mask"].shape)
print("Shape of 'gt_affinities':", f["gt_affinities"].shape)
print("Shape of 'pred_affinities':", f["pred_affinities"].shape)


# %%

# Plotting the raw, GT, affinities, and fg_mask


# Select an element from the batch and set a z-slice
batch_idx = 2
z_idx = 5


fig, axes = plt.subplots(5, 4, figsize=(20, 20), sharex=True, sharey=True)

# Plot raw in the first row
axes[0][0].imshow(cropped_raw[batch_idx, z_idx], cmap="viridis")
axes[0][0].set_title("Raw")
# Empty plots for rest of the columns in first row
for ax in axes[0][1:]:
    ax.axis("off")

# Plot gt in the second row
axes[1][0].imshow(f["gt"][batch_idx, z_idx], cmap="viridis")
axes[1][0].set_title("gt")
# Empty plots for rest of the columns in second row
for ax in axes[1][1:]:
    ax.axis("off")

# Plot fg_mask in the third row
axes[2][0].imshow(f["fg_mask"][batch_idx, z_idx], cmap="viridis")
axes[2][0].set_title("fg_mask")
# Empty plots for rest of the columns in third row
for ax in axes[2][1:]:
    ax.axis("off")

# Plot each channel of gt_affinities in the fourth row
for i, ax in enumerate(axes[3]):
    if i == 0:
        ax.imshow(
            np.sum(f["gt_affinities"][batch_idx, :, z_idx], axis=0), cmap="viridis"
        )
        ax.set_title("gt_affinities summed")
    else:
        ax.imshow(f["gt_affinities"][batch_idx, i - 1, z_idx], cmap="viridis")
        ax.set_title(f"gt_affinities channel {i-1}")

# Plot each channel of pred_affinities in the fifth row
for i, ax in enumerate(axes[4]):
    if i == 0:
        ax.imshow(
            np.sum(f["pred_affinities"][batch_idx, :, z_idx], axis=0), cmap="viridis"
        )
        ax.set_title("pred_affinities summed")
    else:
        ax.imshow(f["pred_affinities"][batch_idx, i - 1, z_idx], cmap="viridis")
        ax.set_title(f"pred_affinities channel {i-1}")

plt.tight_layout()
plt.show()

# %%

# Define watershed and segmentation


def watershed_from_boundary_distance(
    boundary_distances, boundary_mask, id_offset=0, min_seed_distance=10
):
    # Use a 3D kernel for maximum filtering in 3D
    kernel = (min_seed_distance, min_seed_distance, min_seed_distance)
    max_filtered = maximum_filter(boundary_distances, size=kernel)
    maxima = max_filtered == boundary_distances
    seeds, n = label(maxima)

    print(f"Found {n} fragments")

    if n == 0:
        return np.zeros_like(boundary_distances, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    # 3D watershed
    fragments = watershed(
        boundary_distances.max() - boundary_distances, markers=seeds, mask=boundary_mask
    )

    return fragments.astype(np.uint64), n + id_offset


def watershed_from_affinities(
    affs, max_affinity_value=1.0, id_offset=0, min_seed_distance=10
):
    # Adjust for 3D
    mean_affs = np.mean(affs[1:], axis=0)
    boundary_mask = mean_affs > 0.5 * max_affinity_value
    boundary_distances = distance_transform_edt(boundary_mask)

    return watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        id_offset=id_offset,
        min_seed_distance=min_seed_distance,
    )


def get_segmentation(affinities, threshold):
    fragments = watershed_from_affinities(affinities)[0]
    thresholds = [threshold]

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),
        fragments=fragments,
        thresholds=thresholds,
    )

    return next(generator)


# %%

# # Make segmentations
# pred_affs = f["pred_affinities"][batch_idx]

# ws_affs = np.stack([pred_affs[0], pred_affs[1], pred_affs[2]], axis=0)

# # threshold = 0.8
# # segmentation = get_segmentation(ws_affs, threshold)


# # %%

# # Print shapes
# print("Shape of 'raw':", f["raw"].shape)
# print("Shape of 'gt':", f["gt"].shape)
# print("Shape of 'cropped raw':", cropped_raw.shape)


# # %%
# # Plotting the predictions and the segmentation

# z_slice = 2  # example Z slice for visualization

# fig, axes = plt.subplots(4, 4, figsize=(20, 20), sharex=True, sharey=True)


# # 1st row: First, display the resized_raw, then GT displayed 3 times for layout uniformity
# axes[0, 0].imshow(cropped_raw[batch_idx, z_slice], cmap="viridis")
# axes[0, 0].set_title("Raw")

# for ax in axes[0][1:]:
#     ax.imshow(f["gt"][batch_idx, z_slice], cmap="viridis")
#     ax.set_title("GT")

# # 2nd row: GT affinities (3 channels)
# axes[1][0].imshow(f["gt_affinities"][batch_idx, 0, z_slice], cmap="viridis")
# axes[1][0].set_title("GT Affinity 1")
# axes[1][1].imshow(f["gt_affinities"][batch_idx, 1, z_slice], cmap="viridis")
# axes[1][1].set_title("GT Affinity 2")
# axes[1][2].imshow(f["gt_affinities"][batch_idx, 2, z_slice], cmap="viridis")
# axes[1][2].set_title("GT Affinity 3")
# axes[1][3].imshow(
#     np.sum(f["gt_affinities"][batch_idx, :, z_slice], axis=0), cmap="viridis"
# )
# axes[1][3].set_title("GT Affinities Summed")

# # 3rd row: Pred affinities (3 channels)
# axes[2][0].imshow(pred_affs[0, z_slice], cmap="viridis", vmax=0.005)
# axes[2][0].set_title("Pred Affinity 1")
# axes[2][1].imshow(pred_affs[1, z_slice], cmap="viridis", vmax=0.005)
# axes[2][1].set_title("Pred Affinity 2")
# axes[2][2].imshow(pred_affs[2, z_slice], cmap="viridis", vmax=0.005)
# axes[2][2].set_title("Pred Affinity 3")
# axes[2][3].imshow(np.sum(pred_affs[:, z_slice], axis=0), cmap="viridis")
# axes[2][3].set_title("Pred Affinities Summed")

# # 4th row: Segmentation
# axes[3][0].imshow(segmentation[z_slice], cmap="viridis")
# axes[3][0].set_title("Segmentation")
# # Rest of the axes in 4th row are empty (or you can fill them with other data if needed)
# for ax in axes[3][1:]:
#     ax.axis("off")

# plt.tight_layout()
# plt.show()

# %%
# Directory containing zarr files
zarr_directory = "/mnt/efs/shared_data/hack/lsd/aff_LB"

# Get a list of all zarr files in the directory
zarr_files = [f for f in os.listdir(zarr_directory) if f.endswith(".zarr")]

for zarr_file in zarr_files:
    full_path = os.path.join(zarr_directory, zarr_file)

    # Print the name of the zarr file
    print(f"Processing file: {zarr_file}")

    # Open zarr container with read/write permissions
    with zarr.open(full_path, mode="a") as f:
        # Extract the prediction affinities
        pred_affs = f["pred_affs"][:]

        # Stack the affinities
        print(pred_affs.shape)
        pred_affs_rm = pred_affs.squeeze(axis=0)
        print(pred_affs_rm.shape)

        ws_affs = np.stack([pred_affs_rm[0], pred_affs_rm[1], pred_affs_rm[2]], axis=0)

        threshold = 0.5
        segmentation = get_segmentation(ws_affs, threshold)

        # Save segmentation to zarr container
        if "segmentation" in f:
            del f["segmentation"]  # delete the existing dataset if it exists
        f.create_dataset("segmentation", data=segmentation, chunks=True)

# %%
