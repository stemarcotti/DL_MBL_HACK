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
import zarr

from funlib.learn.torch.models import UNet, ConvPass
from gunpowder.torch import Predict

from affs_model import MtlsdModel, WeightedMSELoss, get_model

logging.basicConfig(level=logging.DEBUG)


output_shape = gp.Coordinate((8, 88, 88))
input_shape = gp.Coordinate((24, 128, 128))
voxel_size = gp.Coordinate((250, 75, 75))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

# total roi of image to predict on
total_roi = gp.Coordinate((64, 640, 640))*voxel_size
     

stack_size = 16


# model = get_model()
# loss = WeightedMSELoss()


def predict(checkpoint, raw_file, raw_dataset):
    logging.basicConfig(level=logging.DEBUG)
    raw = gp.ArrayKey("RAW")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_affs, output_size)

    context = (input_size - output_size) / 2

    source = gp.ZarrSource(
        raw_file, {raw: raw_dataset}, {raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size)}
    )

    with gp.build(source):
        # print(source.spec[raw])
        total_input_roi = source.spec[raw].roi
        total_output_roi = source.spec[raw].roi.grow(-context, -context)

    model, _, _ = get_model()

    # set model to eval mode
    model.eval()

    # add a predict node
    predict = gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs={"input": raw},
        outputs={0: pred_affs},
    )

    # this will scan in chunks equal to the input/output sizes of the respective arrays
    scan = gp.Scan(scan_request)

    pipeline = source
    pipeline += gp.Normalize(raw)

    # raw shape = h,w

    pipeline += gp.Unsqueeze([raw])

    # raw shape = c,h,w
    #pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(1)

    # raw shape = b,c,h,w


    pipeline += predict
    pipeline += scan
    # pipeline += gp.Squeeze([raw])

    # # raw shape = c,h,w
    # # pred_lsds shape = b,c,h,w
    # # pred_affs shape = b,c,h,w

    # pipeline += gp.Squeeze([raw, pred_affs])

    # raw shape = h,w
    # pred_lsds shape = c,h,w
    # pred_affs shape = c,h,w

    predict_request = gp.BatchRequest()

    # this lets us know to process the full image. we will scan over it until it is done
    #predict_request.add(raw, input_size)
    # predict_request.add(pred_affs, output_size)
    predict_request.add(raw, total_input_roi.get_end())
    predict_request.add(pred_affs, total_output_roi.get_end())
    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[raw].data, batch[pred_affs].data




def load_ground_truth(raw_file, gt_dataset):
    gt = gp.ArrayKey("GT")
    scan_request = gp.BatchRequest()
    scan_request.add(gt, input_size)

    # Set the zarr source for ground truth
    source = gp.ZarrSource(
        raw_file, 
        {gt: gt_dataset}, 
        {gt: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size)}
    )

    with gp.build(source):
        total_input_roi = source.spec[gt].roi

    pipeline = source
    #pipeline += gp.Normalize(gt)  # Normalize if required; can be removed if not needed

    gt_request = gp.BatchRequest()
    gt_request.add(gt, total_input_roi.get_end())

    with gp.build(pipeline):
        batch = pipeline.request_batch(gt_request)

    return batch[gt].data



#%% 

checkpoint = "/mnt/efs/shared_data/hack/lsd/aff_exp2/just_affs_checkpoint_3300"

def save_to_zarr(data, folder, filename, dataset):
    output_path = os.path.join(folder, filename + ".zarr")
    with zarr.open(output_path, mode='a') as f:
        # Check if dataset exists and delete
        if dataset in f:
            del f[dataset]
        f[dataset] = data



output_folder = "/mnt/efs/shared_data/hack/lsd/aff_LB"


# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

load_path = [
    '/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr',
    '/mnt/efs/shared_data/hack/data/20230504/20230504_raw.zarr',
    '/mnt/efs/shared_data/hack/data/og_cellpose/og_cellpose_raw.zarr'
]
fov_list = [range(5), range(4), range(26)]

for path, fovs in zip(load_path, fov_list):
    for fov in fovs:
        raw_dataset = f"fov{fov}/raw"
        gt_dataset = f"fov{fov}/gt"  # Adjust as per your dataset structure

        raw, pred_affs = predict(checkpoint, path, raw_dataset)
        
        # Assuming you have a method to load GT similar to predict
        gt_data = load_ground_truth(path, gt_dataset)

        # Derive a filename from the path
        filename = os.path.basename(path).replace(".zarr", "") + f"_fov{fov}"

        # Save raw, ground truth, and prediction as zarrs in the output folder
        save_to_zarr(raw, output_folder, filename, "raw")
        save_to_zarr(pred_affs, output_folder, filename, "pred_affs")
        save_to_zarr(gt_data, output_folder, filename, "gt")











# raw_file = "/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr"
# raw_dataset = "fov4/raw"

# raw, pred_affs = predict(checkpoint, raw_file, raw_dataset)

# #%%

# print("Shape of 'raw':", raw.shape)
# print("Shape of 'pred_affs ':", pred_affs.shape)
# %%
# #Plotting of pred affinities

# # Take a slice in the middle along the depth (change index for different slices)
# raw_slice = raw[0, 0, 32, :, :]
# pred_slices = [pred_affs[0, i, 28, :, :] for i in range(3)]
# sum_pred_slice = np.sum(pred_affs[0], axis=0)[28, :, :]

# # Plotting
# fig, axes = plt.subplots(1, 5, figsize=(18, 12))

# axes[0].imshow(raw_slice, cmap='gray')
# axes[0].set_title("Raw")

# for i, pred_slice in enumerate(pred_slices):
#     axes[i+1].imshow(pred_slice, cmap='viridis', vmax=0.01)
#     axes[i+1].set_title(f"Pred Channel {i+1}")

# axes[4].imshow(sum_pred_slice, cmap='viridis', vmax=0.01)
# axes[4].set_title("Sum of Pred")

# for ax in axes:
#     ax.axis('off')

# plt.tight_layout()
# plt.show()



# %%
