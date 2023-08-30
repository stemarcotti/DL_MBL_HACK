#%%
from funlib.learn.torch.models import UNet, ConvPass
import logging
import math
import torch
import multiprocessing
# multiprocessing.set_start_method("fork")
import gunpowder as gp
import zarr
#from iohub import open_ome_zarr
import matplotlib.pyplot as plt
import numpy as np

#%%

#%%
# img_path = '/mnt/efs/shared_data/hack/data/20230811/raw/'
# mask_path = '/mnt/efs/shared_data/hack/data/20230811/fixed_labels/'
store_path = '/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr'

# img_list = os.listdir(img_path)
gt_string = '_deconvolved_rho_0.0038_gamma_0.013_m2_Manual_Mask.tiff'

f = zarr.open(store_path, 'r')
#%%
f['raw'].shape
# f['ground_truth'].shape

#%%
f['raw'].dtype

#%%

# helper function to show image(s), channels first
def imshow(raw, slice_index=0, ground_truth=None, prediction=None):
    num_images = raw.shape[0]
    rows = 1
    if ground_truth is not None:
        rows += 1
    if prediction is not None:
        rows += 1
    cols = num_images

    fig, axes = plt.subplots(rows, cols, figsize=(10, 4), sharex=True, sharey=True, squeeze=False)

    for i in range(num_images):
        # Select the specified slice (e.g., slice_index) for visualization
        selected_slice = raw[i, slice_index]

        axes[0][i].imshow(selected_slice, cmap='gray')  # Assuming grayscale images

    row = 1
    if ground_truth is not None:
        for i in range(num_images):
            axes[row][i].imshow(ground_truth[i][0], cmap='gray')  # Assuming ground_truth[i] is a single channel image
        row += 1

    if prediction is not None:
        for i in range(num_images):
            axes[row][i].imshow(prediction[i][0], cmap='gray')  # Assuming prediction[i] is a single channel image

    plt.show()
#%%
imshow(f['raw'])
# imshow(zarr.open('20230811_raw.zarr')['raw'][:])

# imshow(f, ground_truth=f['ground_truth'][:], prediction=f['ground_truth'][:])
#%%
logging.basicConfig(level=logging.INFO)

num_training_images = 100
batch_size = 32
patch_shape = (252, 252)
learning_rate = 1e-4


def train_until(max_iterations):

    # create model, loss, and optimizer

    unet = UNet(
        in_channels=1,
        num_fmaps=12,
        fmap_inc_factor=4,
        downsample_factors=[(2, 2), (2, 2), (2, 2)],
        kernel_size_down=[[[3, 3], [3, 3]]]*4,
        kernel_size_up=[[[3, 3], [3, 3]]]*3,
        padding='valid',
        constant_upsample=True)
    model = torch.nn.Sequential(
        unet,
        ConvPass(12, 1, [(1, 1)], activation=None),
        torch.nn.Sigmoid())
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # assemble gunpowder training pipeline

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    prediction = gp.ArrayKey('PREDICTION')

    sources = tuple(
        gp.ZarrSource(
            'neurons.zarr',  # the zarr container
            {raw: f'raw_{i}', labels: f'labels_{i}'},  # which dataset to associate to the array key
            {
                raw: gp.ArraySpec(interpolatable=True),
                labels: gp.ArraySpec(interpolatable=False)
            }
        )
        for i in range(num_training_images)
    )
    source = sources + gp.RandomProvider()
    random_location = gp.RandomLocation(
        min_masked=0.1,
        mask=labels)

    normalize_raw = gp.Normalize(raw)
    normalize_labels = gp.Normalize(labels, factor=1.0/255.0)  # ensure labels are float

    simple_augment = gp.SimpleAugment()
    elastic_augment = gp.ElasticAugment(
        control_point_spacing=(16, 16),
        jitter_sigma=(1.0, 1.0),
        rotation_interval=(0, math.pi/2))
    intensity_augment = gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.01,
        shift_max=0.01)
    unsqueeze = gp.Unsqueeze([raw, labels], axis=0)  # add "channel" dim
    stack = gp.Stack(batch_size)
    precache = gp.PreCache(num_workers=4)

    train = gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs = {
          'input': raw
        },
        outputs = {
          0: prediction
        },
        loss_inputs = {
          0: prediction,
          1: labels
        },
        save_every=1000)

    snapshot = gp.Snapshot(
        {
            raw: 'raw',
            prediction: 'prediction',
            labels: 'labels'
        },
        output_filename='iteration_{iteration}.zarr',
        every=100)

    pipeline = (
        source +
        random_location +
        normalize_raw +
        normalize_labels +
        simple_augment +
        elastic_augment +
        intensity_augment +
        unsqueeze +
        stack +
        precache +
        train +
        snapshot)

    output_shape = tuple(model(torch.zeros((1, 1) + patch_shape)).shape[2:])
    print("Input shape:", patch_shape)
    print("Output shape:", output_shape)

    request = gp.BatchRequest()
    request.add(raw, patch_shape)
    request.add(labels, output_shape)
    request.add(prediction, output_shape)

    with gp.build(pipeline):
        for i in range(max_iterations):
            batch = pipeline.request_batch(request)

if __name__ == '__main__':

    train_until(500)