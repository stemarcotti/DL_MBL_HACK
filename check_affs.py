#%%
import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact


#%%
output_folder = "/mnt/efs/shared_data/hack/lsd/aff_LB/"

# Set the path to one of your zarr files (change as needed)
#zarr_file = os.path.join(output_folder, "20230811_fov0.zarr")  #

#zarr_file = os.path.abspath(os.path.join(output_folder, "20230504_fov1.zarr"))

#print(os.listdir(zarr_file))

base_dir = "/mnt/efs/shared_data/hack/lsd/aff_LB"


def load_from_zarr(zarr_file, group_name):
    with zarr.open_group(zarr_file, mode="r") as f:
        data = f[group_name][:]
    return data

# Let's test loading from the zarr directory
zarr_dir = os.path.join(base_dir, "20230504_raw_fov0.zarr")
raw = load_from_zarr(zarr_dir, "raw")

print(raw.shape)

#%%

# Load the data
raw = load_from_zarr(zarr_dir, "raw")
pred_affs = load_from_zarr(zarr_dir, "pred_affs")
gt = load_from_zarr(zarr_dir, "gt")
segmentation = load_from_zarr(zarr_dir, "segmentation")

# Ensure 3D
if raw.ndim == 4:
    raw = raw.squeeze(axis=0)
if pred_affs.ndim == 4:
    pred_affs = pred_affs.squeeze(axis=0)
if gt.ndim == 4:
    gt = gt.squeeze(axis=0)
if segmentation.ndim == 4:
    segmentation = segmentation.squeeze(axis=0)

# Assuming the data shape is (Z, Y, X) for the 3D volume
num_slices = raw.shape[0]

print("Shape of raw:", raw.shape)
print("Shape of pred_affs:", pred_affs.shape)
print("Shape of gt:", gt.shape)
print("Shape of segmentation:", segmentation.shape)

# %%

# Extract middle slices
raw_slice = np.squeeze(raw)[31]         # (640, 640)
pred_aff_slices = pred_affs[0, :, 27]   # (3, 620, 620)
gt_slice = gt[31]                       # (640, 640)
seg_slice = segmentation[31]            # Assuming the same shape as raw (640, 640)

# Plotting

fig, axes = plt.subplots(1, 6, figsize=(30, 5), sharex=True, sharey=True)

# 1. Raw slice
axes[0].imshow(raw_slice, cmap="viridis")
axes[0].set_title("Raw\n" + f"Max: {np.max(raw_slice):.4f}")

# 2. Pred affinities slices
for i in range(3):
    axes[i + 1].imshow(pred_aff_slices[i], cmap="viridis", vmax=1.5)
    axes[i + 1].set_title(f"Pred Affinity {i + 1}\n" + f"Max: {np.max(pred_aff_slices[i]):.4f}")

# 3. GT slice
axes[4].imshow(gt_slice, cmap="viridis")
axes[4].set_title("GT\n" + f"Max: {np.max(gt_slice):.4f}")

# 4. Segmentation slice
axes[5].imshow(seg_slice, cmap="viridis", vmax=0.1)
axes[5].set_title("Segmentation\n" + f"Max: {np.max(seg_slice):.4f}")

plt.tight_layout()
plt.show()

# %%
