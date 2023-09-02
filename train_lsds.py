import zarr
import gunpowder as gp
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging

from funlib.learn.torch.models import UNet, ConvPass
from lsd.train.gp import AddLocalShapeDescriptor
from gunpowder.torch import Train
from torchsummary import summary

logging.basicConfig(level=logging.DEBUG)

class MtlsdModel(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down,
        kernel_size_up,
        constant_upsample):
  
        super().__init__()

        # create unet
        self.unet = UNet(
          in_channels=in_channels,
          num_fmaps=num_fmaps,
          fmap_inc_factor=fmap_inc_factor,
          downsample_factors=downsample_factors,
          kernel_size_down=kernel_size_down,
          kernel_size_up=kernel_size_up,
          constant_upsample=constant_upsample,
          padding="same")

        # create lsd and affs heads
        self.lsd_head = ConvPass(num_fmaps, 10, [[1, 1, 1]], activation='Sigmoid')
        self.aff_head = ConvPass(num_fmaps, 3, [[1, 1, 1]], activation='Sigmoid')

    def forward(self, input):

        # pass raw through unet
        z = self.unet(input)

        # pass output through heads
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)

        return lsds, affs


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def _calc_loss(self, pred, target, weights):

        scaled = weights * (pred - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

    def forward(
        self,
        lsds_prediction,
        lsds_target,
        lsds_weights,
        affs_prediction,
        affs_target,
        affs_weights,
    ):

        # calc each loss and combine
        loss1 = self._calc_loss(lsds_prediction, lsds_target, lsds_weights)
        loss2 = self._calc_loss(affs_prediction, affs_target, affs_weights)

        return loss1 + loss2



def main():
    # Config for data
    load_path = ['/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr',
             '/mnt/efs/shared_data/hack/data/20230504/20230504_raw.zarr']
    fov_list = [[0,1,2,3], [1,2,3]]
    output_shape = gp.Coordinate((16, 256, 256))
    stack_size = 1
    voxels = gp.Coordinate((250, 75, 75))  

    # Array keys
    raw = gp.ArrayKey("RAW")
    gt = gp.ArrayKey("GROUND_TRUTH")
    fg = gp.ArrayKey("FOREGROUND")
    gt_lsds = gp.ArrayKey('GT_LSDS')
    lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    gt_affs = gp.ArrayKey('GT_AFFS')
    affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
    pred_affs = gp.ArrayKey('PRED_AFFS')


    # How many stacks to request
    stack = gp.Stack(stack_size)

    # random sampeling
    random_location = gp.RandomLocation()

    # geometric augmentation
    simple_augment = gp.SimpleAugment(mirror_probs=[0, 0, 0], transpose_probs=[0, 0, 0])

    # elastic_augment = gp.ElasticAugment(
    #     control_point_spacing=(30, 30, 30),
    #     jitter_sigma=(1.0, 1.0, 1.0),
    #     rotation_interval=(0, math.pi / 2),
    #     spatial_dims=3,
    # )

    # signal augmentations
    intensity_augment = gp.IntensityAugment(
        raw, scale_min=0.9, scale_max=1.1, shift_min=-0.01, shift_max=0.01
    )

    noise_augment = gp.NoiseAugment(raw, mode="poisson")

    normalize_raw = gp.Normalize(raw)

    shape_node = AddLocalShapeDescriptor(
        gt,
        gt_lsds,
        lsds_mask=lsds_weights,
        sigma=1500,
        downsample=2)
    
    affinity_node = gp.AddAffinities(
        affinity_neighborhood=[
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, -1]],
        labels=gt,
        affinities=gt_affs,
        dtype=np.float32)
    
    balance_node = gp.BalanceLabels(
        gt_affs,
        affs_weights
    )

    sources = tuple([])
    for i, path in enumerate(load_path):
        print(path)
        fovs = fov_list[i]
        print(f"fovs:{fovs}")
        source = tuple(
            gp.ZarrSource(
                path,
                {raw: f"fov{fov}/raw", gt: f"fov{fov}/gt", fg: f"fov{fov}/fg_mask"},
                {
                    raw: gp.ArraySpec(interpolatable=True, voxel_size=voxels),
                    gt: gp.ArraySpec(interpolatable=False, voxel_size=voxels),
                    fg: gp.ArraySpec(interpolatable=False, voxel_size=voxels),
                },
            )
            for fov in fovs
        )
        source = source
        if i == 0:
            sources = source
        else:
            sources = sources + source



    # Configure model, loss, etc
    model, loss, optimizer = get_model()

    # Assemble pipeline 
    sources += gp.RandomProvider()  
    # Create the pipeline
    pipeline = sources
    pipeline += normalize_raw
    pipeline += random_location
    pipeline += simple_augment
    #pipeline += elastic_augment
    pipeline += intensity_augment
    pipeline += noise_augment
    pipeline += gp.Reject(mask=fg, min_masked=0.1)
    pipeline += gp.GrowBoundary(gt)
    pipeline += shape_node
    pipeline += affinity_node
    pipeline += balance_node
    
    pipeline += gp.Unsqueeze([raw])

    pipeline += stack

    pipeline += Train(
        model,
        loss,
        optimizer,
        inputs={
            'input': raw
        },
        outputs={
            0: pred_lsds,
            1: pred_affs
        },
        loss_inputs={
            0: pred_lsds,
            1: gt_lsds,
            2: lsds_weights,
            3: pred_affs,
            4: gt_affs,
            5: affs_weights
        })


    request = gp.BatchRequest()
    request[raw] = gp.Roi((0, 0, 0), output_shape*voxels)
    request[gt] = gp.Roi((0, 0, 0), output_shape*voxels)
    request[fg] = gp.Roi((0, 0, 0), output_shape*voxels)
    request[gt_lsds]= gp.Roi((0, 0, 0), output_shape*voxels)
    request[lsds_weights]= gp.Roi((0, 0, 0), output_shape*voxels)
    request[pred_lsds]= gp.Roi((0, 0, 0), output_shape*voxels)
    request[gt_affs]= gp.Roi((0, 0, 0), output_shape*voxels)
    request[affs_weights]= gp.Roi((0, 0, 0), output_shape*voxels)
    request[pred_affs]= gp.Roi((0, 0, 0), output_shape*voxels)

    


    # Build pipeline with training loop 
    with gp.build(pipeline):
        batch = pipeline.request_batch(request)


def get_model():
    # Configure model, loss, etc
    in_channels=1
    num_fmaps=12
    fmap_inc_factor=5
    ds_fact = [(2,2,2),(2,2,2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3,3,3), (3,3,3)]]*num_levels
    ksu = [[(3,3,3), (3,3,3)]]*(num_levels - 1)
    constant_upsample = True

    model = MtlsdModel(
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        ds_fact,
        ksd,
        ksu,
        constant_upsample)
    
    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(lr=0.5e-4, params=model.parameters())

    return model, loss, optimizer

if __name__ == "__main__":
    main()
    #model, loss, optimizer = get_model()
    #summary(model, (1, 16,  256, 256))

