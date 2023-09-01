import zarr
import gunpowder as gp
import math
import numpy as np
import matplotlib.pyplot as plt
import torch


def process_zarr_data(load_path, output_shape=(128, 128), device='cuda'):
    # Open the zarr file
    f = zarr.open(load_path, 'r')

    # Define keys
    raw = gp.ArrayKey('RAW')
    gt = gp.ArrayKey('GROUND_TRUTH')

    #Define paramters
    random_location = gp.RandomLocation()
    simple_augment = gp.SimpleAugment()
    stack = gp.Stack(5)

    elastic_augment = gp.ElasticAugment(
    control_point_spacing=(20, 20),
    jitter_sigma=(1.0, 1.0),
    rotation_interval=(0, math.pi/2))

    normalize = gp.Normalize(raw)
    intensity_augment = gp.IntensityAugment(
    raw,
    scale_min=0.8,
    scale_max=1.2,
    shift_min=-0.2,
    shift_max=0.2)
    noise_augment = gp.NoiseAugment(raw)

    # Gunpowder pipeline
    source = tuple(gp.ZarrSource(
    load_path,
    {
      raw: f'fov{i}/raw',
      gt: f'fov{i}/gt'
    },
    {
      raw: gp.ArraySpec(interpolatable=True,
        voxel_size=(1,1)),
      gt: gp.ArraySpec(interpolatable=False,
        voxel_size=(1,1)),
    }) + normalize + random_location for i in [1]) 

    normalize = gp.Normalize(raw)
    pad_raw = gp.Pad(raw, None)
    pad_gt = gp.Pad(gt, 0)

    # Create the pipeline
    pipeline = source
    pipeline += gp.RandomProvider()
    pipeline += simple_augment
    pipeline += elastic_augment
    pipeline += stack
    pipeline += intensity_augment
    pipeline += noise_augment
    

    # Define a batch request
    request = gp.BatchRequest()
    request[raw] = gp.Roi((0, 0), output_shape)
    request[gt] = gp.Roi((0, 0), output_shape)

    # Build the pipeline and request the batch
    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    # Move the batch data to the specified device
    if device == 'cuda':
        batch[raw].data = torch.from_numpy(batch[raw].data).cuda()
        batch[gt].data = torch.from_numpy(batch[gt].data).cuda()

    return batch

# Example usage:
load_path = '/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr'
output_shape = (128, 128)
device = 'cuda'  # Change to 'cpu' if you want to use CPU
result_batch = process_zarr_data(load_path, output_shape, device)

#print(result_batch[raw].data.shape)
#print(result_batch[gt].data.shape)

   # Print the shape of the 'raw' data
print(result_batch[gp.ArrayKey('RAW')].data.shape)
print(result_batch[gp.ArrayKey('GROUND_TRUTH')].data.shape)
