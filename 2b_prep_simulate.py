"""Prepare mri_simulate outputs for training.

Run mri_simulate with `simu.affine = 1` and `simu.clean = 1`. It then already
writes the T1w and its ground truth on the 336x384x336 grid of 0.5mm voxels
that 2_prep_segment.py produces, so none of the CAT12 preprocessing is needed
here: no second segmentation, no affine, no bias correction. This script only
skull-strips, normalizes, derives the 0.75mm versions and pairs the files.

Several T1w images share one label - runs that differ only in noise, bias field
or motion are the same anatomy - and each of them becomes one training sample.
The pairing follows the BIDS entities that mri_simulate writes: the label
carries only the anatomical tags in its desc, the image carries the acquisition
tags in front of them.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from niftiai import TensorImage3d
from spline_resize import resize
from src.synth import tissue_weights
from src.bids import pair_simulations
ALIGN = True
data_path = 'data'
simu_path = 'data/simu'   # the mri_simulate derivatives folder
n_folds = 5
nogm_threshold = .015     # same as in 2_prep_segment.py


def min_max(x, low=.005, high=.995):  # same normalization as 2_prep_segment.py
    mask = x > 0
    low, high = np.percentile(x[mask].cpu(), 100 * low), np.percentile(x[mask].cpu(), 100 * high)
    x = (x - low) / (high - low)
    x[x > 1] = 1 + torch.log10(x[x > 1])
    return x.clamp(min=0)


if __name__ == '__main__':
    shape_05mm = (336, 384, 336)
    shape_075mm = (224, 256, 224)
    nib_affine_05mm = np.array([[.5, 0, 0, -84], [0, .5, 0, -120], [0, 0, .5, -72], [0, 0, 0, 0]])
    nib_affine_075mm = np.array([[.75, 0, 0, -84], [0, .75, 0, -120], [0, 0, .75, -72], [0, 0, 0, 0]])
    subdirs = ['img_05mm_minmax', 'img_075mm_minmax', 'p0_05mm', 'p0_075mm', 'nogm', 'csvs']
    for subdir in subdirs: Path(f'{data_path}/{subdir}').mkdir(parents=True, exist_ok=True)

    df = pair_simulations(simu_path, n_folds)
    if len(df) == 0:
        raise SystemExit(f'No simulated T1w/dseg pairs found below {simu_path}')
    print(f'{len(df)} simulations from {df.subject.nunique()} subjects, '
          f'{(df.gm_probseg != "").sum()} of them with a GM probseg (needed for nogm)')

    for row in tqdm(df.itertuples(), total=len(df)):
        im = TensorImage3d.create(row.t1w).cuda()
        p0 = TensorImage3d.create(row.dseg).cuda()
        if tuple(im.shape[-3:]) != shape_05mm:
            raise ValueError(f'{row.t1w} has shape {tuple(im.shape[-3:])}, expected {shape_05mm}. '
                             f'Run mri_simulate with simu.affine = 1.')
        header = im.header
        header.set_data_dtype(np.float32)
        p0_header = p0.header

        im[p0 <= 0] = 0  # brain extraction, the label is its own mask
        im = min_max(im)
        TensorImage3d(im, affine=nib_affine_05mm, header=header).save(
            f'{data_path}/img_05mm_minmax/{row.filename}.nii.gz')
        TensorImage3d(p0, affine=nib_affine_05mm, header=p0_header).save(
            f'{data_path}/p0_05mm/{row.filename}.nii.gz')

        # nogm: where the triangular decomposition of the label overestimates GM.
        # With the exact GM fraction of the simulation this is the ground truth
        # that 2_prep_segment.py can only estimate from the CAT12 p1.
        if row.gm_probseg:
            p1 = TensorImage3d.create(row.gm_probseg).cuda()
            nogm = (tissue_weights(p0[None].float())[:, 1] - p1) > nogm_threshold
            TensorImage3d(nogm.float(), affine=nib_affine_05mm, header=header).save(
                f'{data_path}/nogm/{row.filename}.nii.gz')

        im = resize(im[None], shape_075mm, align_corners=ALIGN, mask_value=0)[0]
        TensorImage3d(im, affine=nib_affine_075mm, header=header).save(
            f'{data_path}/img_075mm_minmax/{row.filename}.nii.gz')
        p0 = resize(p0[None], shape_075mm, align_corners=ALIGN, mask_value=0)[0]
        TensorImage3d(p0, affine=nib_affine_075mm, header=p0_header).save(
            f'{data_path}/p0_075mm/{row.filename}.nii.gz')

    csv_path = f'{data_path}/csvs/simulated.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nWrote {csv_path}. Set csv_name to "simulated.csv" (and eval_suffix to "")'
          f' at the top of 3_train_segment.py, 4_train_segment_patches.py and 5_train_nogm.py.')
    print('To train on real and simulated data together, concatenate the two csvs '
          'and keep the fold column, which follows the source subject.')
