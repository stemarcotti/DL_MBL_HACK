import zarr
import gunpowder as gp
import math
import numpy as np
import matplotlib.pyplot as plt
import torch


def prepare_gunpowder_pipeline(load_path, output_shape=(30, 128, 128), device='cuda'): #stack_size=5
 
    # Define keys
    raw = gp.ArrayKey('RAW')
    gt = gp.ArrayKey('GROUND_TRUTH')

    ########  Define paramters ########

    # How many stacks to request
    #stack = gp.Stack(stack_size)

    # random sampeling 
    random_location = gp.RandomLocation()

    # geometric augmentation
    simple_augment = gp.SimpleAugment(
        mirror_probs = [0,0,0],
        transpose_probs = [0,0,0])
    
    elastic_augment = gp.ElasticAugment(
        control_point_spacing=(30, 30, 30),
        jitter_sigma=(1.0, 1.0, 1.0),
        rotation_interval=(0, math.pi/2),
        spatial_dims = 3
    )

    # signal augmentations
    intensity_augment = gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.01,
        shift_max=0.01)

    noise_augment = gp.NoiseAugment(raw, mode='poisson')

    normalize_raw = gp.Normalize(raw)
    


    ##### Gunpowder pipeline #####

    # Load the data
    source = tuple(gp.ZarrSource(
    load_path,
    {
      raw: f'fov{i}/raw',
      gt: f'fov{i}/gt'
    },
    {
      raw: gp.ArraySpec(interpolatable=True,
        voxel_size=(1,1,1)),
      gt: gp.ArraySpec(interpolatable=False,
        voxel_size=(1,1,1)),
    })  + normalize_raw + random_location for i in [1]) 


    # Create the pipeline
    pipeline = source
    pipeline += gp.RandomProvider()
    pipeline += simple_augment
    pipeline += elastic_augment
    # pipeline += stack
    pipeline += intensity_augment
    pipeline += noise_augment
    
    # Define a batch request
    request = gp.BatchRequest()
    request[raw] = gp.Roi((0, 0, 0), output_shape)
    request[gt] = gp.Roi((0, 0, 0), output_shape)

    return pipeline, request
    



# # Build the pipeline and request the batch
# with gp.build(pipeline):
#   batch = pipeline.request_batch(request)


# # Move the batch data to the specified device
# if device == 'cuda':
#     batch[raw].data = torch.from_numpy(batch[raw].data).cuda()
#     batch[gt].data = torch.from_numpy(batch[gt].data).cuda()

# Example usage:
#load_path = '/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr'
#output_shape = (15, 128, 128)
#stack_size = 7  # Adjust the stack size as needed
#device = 'cuda'  # Change to 'cpu' if you want to use CPU
#result_batch = process_zarr_data(load_path, output_shape, stack_size, device)

#print(result_batch[raw].data.shape)
#print(result_batch[gt].data.shape)

   # Print the shape of the 'raw' data
#print("Raw shape: " + str(result_batch[gp.ArrayKey('RAW')].data.shape))
#print("Ground truth shape: " + str(result_batch[gp.ArrayKey('GROUND_TRUTH')].data.shape))
