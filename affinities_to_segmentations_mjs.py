
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
        markers=seeds, #was seeds
        mask=boundary_mask)

    return fragments.astype(np.uint64), n + id_offset #ret 

#%%

def watershed_from_affinities(
        affs,
        max_affinity_value=1.0,
        id_offset=0,
        min_seed_distance=10):

    mean_affs = 0.5 * (affs[0] + affs[1] + affs[2])
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
load_path = "/mnt/efs/shared_data/hack/lsd/mjs_onlyaffs_norm3000_exp5/snapshot"

# List all Zarr files in the folder
zarr_files = [file for file in os.listdir(load_path) if file.endswith('.zarr')]
zarr_file = zarr_files[0]

zarr_path = os.path.join(load_path, zarr_file)
zarr_snap = zarr.open(zarr_path, mode='r')

segmentations = []
# affinities = 'pred_affinities'
chosen_affinities = 'pred_affinities'
batchnumber = 1

print(f'This is the shape of zarr_snap: {zarr_snap[chosen_affinities].shape}')

pred_affs = zarr_snap[chosen_affinities][batchnumber]
ws_affs = np.stack([
pred_affs[0],
pred_affs[1],
pred_affs[2]]
)
print(f'This is the shape of pred_affs: {pred_affs.shape}')

print(f'This is the shape of ws_affs: {ws_affs.shape}')
threshold = 0.5
segmentation = get_segmentation(ws_affs, threshold)

#%%


if __name__ == "__main__":

    affs_zarr = sys.argv[1]
    affs_ds = sys.argv[2]
    try:
        thresh = float(sys.argv[3])
    except:
        thresh = 0.3

    affs = zarr.open(affs_zarr,"r")[affs_ds][:]
    affs = (affs/255.0).astype(np.float32)

    fragments = watershed_from_affinities(affs,max_affinity_value=1, fragments_in_xy=True)[0]

    # agglomerate
    max_thresh = 1.0
    step = 1/20

#    if thresholds is None:
#        thresholds = [round(x,2) for x in np.arange(0,max_thresh,step)]
    thresholds = [thresh]

    segs = {}

    generator = waterz.agglomerate(
            affs,
            thresholds=thresholds,
            fragments=fragments.copy())
            #scoring_function=waterz_merge_function[merge_function])

    for threshold,segmentation in zip(thresholds,generator):

        segs[threshold] = segmentation.copy()

    f = zarr.open(affs_zarr,"a")
    f[f'{affs_ds}_watershed_labels_{thresh}'] = segs[thresh]
    f[f'{affs_ds}_watershed_labels_{thresh}'].attrs["offset"] = [0,0,0]
    f[f'{affs_ds}_watershed_labels_{thresh}'].attrs["resolution"] = [70,24,24]


