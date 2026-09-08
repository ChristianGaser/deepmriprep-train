"""Train the two deepbet brain extraction models on simulated data.

Why this is its own script and not part of 2_prep_segment.py: deepbet works on
the *whole head in native space*, while everything else here works on the
336x384x336 grid of 0.5mm voxels, whose 168x192x168mm field of view slices
straight through the skull. Run mri_simulate with `simu.affine = 0` for this,
and point simu_path at those outputs. The `_dseg` then comes out on the same
native grid as the image, so `dseg > 0` is the brain mask.

deepbet is a two stage cascade and both stages are reproduced here:

  1. bbox model : the whole volume resampled to 128^3, whose mask gives the
                  bounding box of the brain plus a 10% margin
  2. main model : the volume cropped to that box, resampled to 256^3

Both are resolution agnostic because they resample, so the native voxel size of
the input does not matter.

Note what this changes: the mask is CAT12's APRG mask, so the model reproduces
that and not the mask deepbet ships. That is the point - 2_prep_segment.py
strips with `p0 > 0` while inference strips with deepbet, and training the
extractor on the same mask the segmentation was trained behind removes that
mismatch. Out of domain it will be less robust than the shipped model, which
saw far more heterogeneous data.
"""
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from fastai.basics import set_seed, Learner
from niftiai import aug_transforms3d, SegmentationDataLoaders3d
from deepbet.bet import BrainExtraction
from src.bids import pair_simulations
from src.loss import DiceFocalLoss
from src.models import Unet3d
from src.transforms import BetNormalize
path = '.'  # if data_path is absolute(=starts with "/") set path = '/'
data_path = 'data'
simu_path = 'data/simu_native'  # mri_simulate outputs of a run with simu.affine = 0
n_folds = 5
SMALL_SHAPE = (128, 128, 128)   # deepbet bbox stage
LARGE_SHAPE = (256, 256, 256)   # deepbet main stage
BBOX_MARGIN = .1                # deepbet default
DIRS = {'small': 'bet_small', 'small_mask': 'bet_small_mask',
        'large': 'bet_large', 'large_mask': 'bet_large_mask'}


def get_bbox_with_margin(mask_small, shape, margin):
    """Mirror of BrainExtraction.get_bbox_with_margin, so that the crop the
    model is trained on is the crop it gets at inference. get_bbox itself is
    imported, being the part where a deviation would be hardest to notice."""
    margin = margin * torch.ones(3)
    scale_factor = torch.tensor(shape) / torch.tensor(mask_small.shape)
    center, size = BrainExtraction.get_bbox(mask_small)
    center, size = scale_factor * center, scale_factor * size
    size = (1 + 2 * margin) * size
    center, size = center.round(), size.round()
    bbox = [[int(c - s / 2), int(c + s / 2)] for c, s in zip(center, size)]
    return tuple([slice(max(0, b[0]), min(s, b[1]), 1) for b, s in zip(bbox, shape)])


def save(vol, fov_mm, fp):
    """The affine only records the effective voxel size of the resampled grid.
    Training reads the arrays, and deepbet maps its output back through the
    bounding box, so nothing downstream depends on it."""
    vx = np.array(fov_mm) / np.array(vol.shape)
    nib.Nifti1Image(np.asarray(vol, dtype=np.float32), np.diag([*vx, 1.])).to_filename(fp)


def prepare(df):
    """Write the fixed size volumes of both stages.

    Only the percentile clip of deepbet's normalize is applied here, because it
    needs the percentiles of the whole head. The z-score and the rescaling that
    follow it are BetNormalize, which runs after the augmentation so the network
    sees the distribution it will see at inference.
    """
    fracs = []
    for row in tqdm(df.itertuples(), total=len(df)):
        out = {k: f'{data_path}/{d}/{row.filename}.nii.gz' for k, d in DIRS.items()}
        img = nib.as_closest_canonical(nib.load(row.t1w))
        msk = nib.as_closest_canonical(nib.load(row.dseg))
        x = torch.from_numpy(np.nan_to_num(img.get_fdata(dtype=np.float32)))
        m = (torch.from_numpy(msk.get_fdata(dtype=np.float32)) > 0).float()
        if x.shape != m.shape:
            raise ValueError(f'{row.t1w} and {row.dseg} differ in shape, '
                             f'{tuple(x.shape)} vs {tuple(m.shape)}. Run mri_simulate '
                             f'with simu.affine = 0 so both stay on the native grid.')
        fov = np.array(img.header.get_zooms()[:3], dtype=np.float64) * np.array(x.shape)

        x_small = F.interpolate(x[None, None], SMALL_SHAPE, mode='nearest-exact')[0, 0]
        m_small = F.interpolate(m[None, None], SMALL_SHAPE, mode='nearest-exact')[0, 0]
        low, high = x_small.quantile(.005), x_small.quantile(.995)
        clip = lambda v: ((v - low) / (high - low)).clamp(0, 1)
        save(clip(x_small), fov, out['small'])
        save(m_small, fov, out['small_mask'])

        # the box comes from the ground truth mask here and from the bbox model
        # at inference, so the main stage is augmented with zoom and translation
        # to cover the difference
        bbox = get_bbox_with_margin(m_small, x.shape, BBOX_MARGIN)
        fov_bbox = np.array([(s.stop - s.start) for s in bbox]) * fov / np.array(x.shape)
        x_large = F.interpolate(x[bbox][None, None], LARGE_SHAPE, mode='nearest-exact')[0, 0]
        m_large = F.interpolate(m[bbox][None, None], LARGE_SHAPE, mode='nearest-exact')[0, 0]
        save(clip(x_large), fov_bbox, out['large'])
        save(m_large, fov_bbox, out['large_mask'])
        fracs.append((float(m_small.mean()), float(m_large.mean())))
    return np.array(fracs).mean(axis=0)


def train_stage(df, img_dir, mask_dir, n_ch, cls_props, batch_tfms, epochs, lr, model_name, bs=1):
    df = df.copy()
    df['img'] = f'{data_path}/{img_dir}/' + df.filename + '.nii.gz'
    df['mask'] = f'{data_path}/{mask_dir}/' + df.filename + '.nii.gz'
    loss_func = DiceFocalLoss(cls_props=cls_props, lambda_focal=1., lambda_gdl=1.)

    # train on the full dataset, one row duplicated as a dummy validation set
    df_total = df.copy()
    df_total.loc[len(df_total)] = df_total.loc[0]
    df_total['is_valid'] = (len(df_total) - 1) * [0] + [1]
    dls = SegmentationDataLoaders3d.from_df(df_total, path=path, fn_col='img', label_col='mask',
                                            valid_col='is_valid', bs=bs, batch_tfms=batch_tfms)
    learn = Learner(dls, model=Unet3d(n_in=1, n_out=2, n_ch=n_ch), loss_func=loss_func)
    learn.model = learn.model.cuda()
    learn.fit_one_cycle(epochs, lr)
    torch.save(learn.model.state_dict(), f'{data_path}/models/{model_name}.pth')

    # cross validation
    for fold in range(n_folds):
        set_seed(1)
        df['is_valid'] = df.fold == fold
        dls = SegmentationDataLoaders3d.from_df(df, path=path, fn_col='img', label_col='mask',
                                                valid_col='is_valid', bs=bs, batch_tfms=batch_tfms)
        learn = Learner(dls, model=Unet3d(n_in=1, n_out=2, n_ch=n_ch), loss_func=loss_func)
        learn.model = learn.model.cuda()
        learn.fit_one_cycle(epochs, lr)
        torch.save(learn.model.state_dict(), f'{data_path}/models/{model_name}_fold{fold}.pth')


if __name__ == '__main__':
    set_seed(1)
    for d in [*DIRS.values(), 'models', 'csvs']:
        Path(f'{data_path}/{d}').mkdir(parents=True, exist_ok=True)

    df = pair_simulations(simu_path, n_folds)
    if len(df) == 0:
        raise SystemExit(f'No simulated T1w/dseg pairs found below {simu_path}')
    print(f'{len(df)} simulations from {df.subject.nunique()} subjects')
    frac_small, frac_large = prepare(df)
    df.to_csv(f'{data_path}/csvs/bet.csv', index=False)
    print(f'brain fraction: {frac_small:.3f} of the whole head, {frac_large:.3f} inside the box')

    # The whole head is the task here, so unlike the segmentation scripts the
    # affine augmentation is switched on: a native T1w comes in any pose, and
    # the bbox model has to find the brain in all of them.
    small_tfms = aug_transforms3d(max_warp=0, max_zoom=.1, max_rotate=.15, max_shear=.02,
                                  max_translate=.05, p_affine=.5, max_ghost=.5, max_spike=2.,
                                  max_bias=.3, max_motion=.5, max_noise=.02, max_down=2,
                                  max_ring=1., max_contrast=.2, max_dof_noise=3, p_flip=.5,
                                  image_mode='nearest', dims_ghost=(0, 1, 2), n_ghosts=2,
                                  p_spike=.1, freq_spike=.5, dims_ring=(0, 1, 2))
    small_tfms += [BetNormalize()]

    # the box of the main stage comes from the bbox model at inference, so zoom
    # and translation stand in for the error that model makes
    large_tfms = aug_transforms3d(max_warp=0, max_zoom=.08, max_rotate=.05, max_shear=0,
                                  max_translate=.05, p_affine=.6, max_ghost=.5, max_spike=2.,
                                  max_bias=.3, max_motion=.5, max_noise=.02, max_down=2,
                                  max_ring=1., max_contrast=.2, max_dof_noise=3, p_flip=.5,
                                  image_mode='nearest', dims_ghost=(0, 1, 2), n_ghosts=2,
                                  p_spike=.1, freq_spike=.5, dims_ring=(0, 1, 2))
    large_tfms += [BetNormalize()]

    train_stage(df, DIRS['small'], DIRS['small_mask'], n_ch=16,
                cls_props=[1 - frac_small, frac_small], batch_tfms=small_tfms,
                epochs=40, lr=1e-3, model_name='brain_extraction_bbox_model', bs=2)
    train_stage(df, DIRS['large'], DIRS['large_mask'], n_ch=8,
                cls_props=[1 - frac_large, frac_large], batch_tfms=large_tfms,
                epochs=30, lr=1e-3, model_name='brain_extraction_model', bs=1)
    print('\nRun 7_compile_models.py, then point deepmriprep at the two .pt files:\n'
          "  run_preprocess(..., bet_model_paths={'model_path': '.../brain_extraction_model.pt',\n"
          "                                       'bbox_model_path': '.../brain_extraction_bbox_model.pt'})")
