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
#%%

#@title utility function to view labels

# matplotlib uses a default shader
# we need to recolor as unique objects

def create_lut(labels):

    max_label = np.max(labels)

    lut = np.random.randint(
            low=0,
            high=255,
            size=(int(max_label + 1), 3),
            dtype=np.uint8)

    lut = np.append(
            lut,
            np.zeros(
                (int(max_label + 1), 1),
                dtype=np.uint8) + 255,
            axis=1)

    lut[0] = 0
    colored_labels = lut[labels]

    return colored_labels

#%%

output_shape = gp.Coordinate((20, 128, 128))
input_shape = gp.Coordinate((20, 128, 128))
voxel_size = gp.Coordinate((250, 75, 75))  
voxels = gp.Coordinate((250, 75, 75)) 
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

# total roi of image to predict on
total_roi = gp.Coordinate((64, 640, 640))*voxels
print('Total ROI:')
print(total_roi)

stack_size = 16



def predict(checkpoint, raw_file, raw_dataset):
    logging.basicConfig(level=logging.DEBUG)
    raw = gp.ArrayKey("RAW")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_affs, output_size)

    context = (output_size) / 2 # (input_size - output_size) / 2

    source = gp.ZarrSource(
        raw_file, {raw: raw_dataset},
          {raw: gp.ArraySpec(interpolatable=True, voxel_size=voxels)}
    )

    with gp.build(source):
        # print(source.spec[raw])
        total_input_roi = source.spec[raw].roi
        total_output_roi = source.spec[raw].roi #.grow(-context, -context)

    model, _, _ = get_model() ## This is different...

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
    pipeline += gp.Normalize(raw, factor = 1/3000) #factor = 1/3000

    # raw shape = h,w

    pipeline += gp.Unsqueeze([raw])

    # raw shape = c,h,w
    #pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(1) #should this be stack_size?

    # raw shape = b,c,h,w


    pipeline += predict
    pipeline += scan
    # pipeline += gp.Squeeze([raw])

    # # raw shape = c,h,w
    # # pred_lsds shape = b,c,h,w
    # # pred_affs shape = b,c,h,w

    pipeline += gp.Squeeze([raw, pred_affs])

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
        # print(batch[raw].data.shape)

    return batch[raw].data, batch[pred_affs].data


#%% 
checkpoint = "/mnt/efs/shared_data/hack/lsd/mjs_onlyaffs_norm3000_exp5/first_try_checkpoint_2100"
raw_file = "/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr"
raw_dataset = "fov2/raw"

raw, pred_affs = predict(checkpoint, raw_file, raw_dataset)

#%%


fig, axes = plt.subplots(
            1,
            2,
            figsize=(20, 6),
            sharex=True,
            sharey=True,
            squeeze=False)

# view predictions (for lsds we will just view the mean offset component)
axes[0][0].imshow(raw, cmap='gray')
axes[0][1].imshow(np.squeeze(pred_affs[0]), cmap='jet')

# %%
