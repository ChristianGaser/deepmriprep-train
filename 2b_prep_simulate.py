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
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from niftiai import TensorImage3d
from spline_resize import resize
from src.synth import tissue_weights
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


def split_entities(path):
    """Split a BIDS filename into everything before the desc entity, the desc
    label and the suffix. `_label-GM_probseg` keeps its label entity in the
    suffix part, which is what pairs it with its dseg."""
    name = Path(path).name
    for ext in ('.nii.gz', '.nii'):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    m = re.match(r'^(?P<base>.*?)(?:_desc-(?P<desc>[A-Za-z0-9]+))?_(?P<suffix>[A-Za-z0-9-]+)$', name)
    return m.group('base'), m.group('desc') or '', m.group('suffix')


def pair_simulations(simu_dir):
    """Pair every simulated T1w with the label that belongs to its anatomy.

    A label desc holds the anatomical tags plus `Clean`, an image desc holds the
    acquisition tags followed by the same anatomical tags. The label whose
    anatomical part is the longest suffix of the image desc is therefore the
    right one, which keeps a `Wmh2` image away from the label of a run without
    WMHs. A cleaned label wins over an uncleaned one of the same anatomy.
    """
    labels = {}
    for fp in sorted(Path(simu_dir).rglob('*_dseg.nii*')):
        base, desc, _ = split_entities(fp)
        anat = desc[:-5] if desc.endswith('Clean') else desc
        labels.setdefault(base, []).append((anat, desc.endswith('Clean'), fp))

    rows = []
    for fp in sorted(Path(simu_dir).rglob('*_T1w.nii*')):
        base, desc, _ = split_entities(fp)
        if 'Biasfield' in desc:
            continue
        cands = [c for c in labels.get(base, []) if desc.endswith(c[0])]
        if not cands:
            print(f'No label found for {fp.name}, skipped.')
            continue
        anat, is_clean, lab = max(cands, key=lambda c: (len(c[0]), c[1]))
        gm = Path(str(lab).replace('_dseg.nii', '_label-GM_probseg.nii'))
        rows.append({'filename': f'{base}_desc-{desc}' if desc else base,
                     'subject': base.split('_space-')[0].split('_res-')[0],
                     't1w': str(fp), 'dseg': str(lab),
                     'gm_probseg': str(gm) if gm.exists() else ''})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    # the fold has to follow the source subject, otherwise variants of one brain
    # end up on both sides of the cross validation
    subjects = sorted(df.subject.unique())
    fold = {s: i % n_folds for i, s in enumerate(subjects)}
    df['fold'] = df.subject.map(fold)
    return df


if __name__ == '__main__':
    shape_05mm = (336, 384, 336)
    shape_075mm = (224, 256, 224)
    nib_affine_05mm = np.array([[.5, 0, 0, -84], [0, .5, 0, -120], [0, 0, .5, -72], [0, 0, 0, 0]])
    nib_affine_075mm = np.array([[.75, 0, 0, -84], [0, .75, 0, -120], [0, 0, .75, -72], [0, 0, 0, 0]])
    subdirs = ['img_05mm_minmax', 'img_075mm_minmax', 'p0_05mm', 'p0_075mm', 'nogm', 'csvs']
    for subdir in subdirs: Path(f'{data_path}/{subdir}').mkdir(parents=True, exist_ok=True)

    df = pair_simulations(simu_path)
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
