[![DOI](https://zenodo.org/badge/1011143977.svg)](https://doi.org/10.5281/zenodo.17749044)

**Disclaimer**: deepmriprep is not related to fMRIPrep or sMRIPrep and is not part of the NiPreps framework

![logo](https://github.com/user-attachments/assets/bbd01efd-ba71-4504-a085-909b28366de4)
 
This repo contains scripts to train neural networks used in [deepmriprep](https://github.com/wwu-mmll/deepmriprep)

## Installation 🛠️
Install CAT12 ([version 12.8.2](https://github.com/ChristianGaser/cat12/releases/tag/12.8.2) was used in [the publication](https://arxiv.org/abs/2408.10656)) 

To install pycairo (dependency of [niftiai](https://github.com/codingfisch/niftiai)) run 

`sudo apt install libcairo2-dev pkg-config python3-dev`

`torch` with (your version of) CUDA support should be installed first via the [proper install command](https://pytorch.org/get-started/locally)

Then use `requirements.txt` to install the remaining dependencies

## Download MRIs 📥
Pick a folder on a fast disk(=SSD) on your system with ~500GB of free space (per default `data`)

If that folder should not be `data` (the default), 
1. copy and paste the `data` folder into your desired folder
2. adapt `data_path` (and `path`, see comment) at the beginning of each of the 7 scripts

Download the T1w MRIs listed in `data/csvs/openneuro_hd.csv` from [OpenNeuro](https://openneuro.org/)

## Preprocess
The downloaded MRIs should be placed in `data/t1` with the filenames from `openneuro_hd.csv`

Applying CAT12 to these MRIs should—e.g., for the `p0` output of filename `0009_sub-06`—result in 
```
data/t1/CAT12.8.2/mri/p00009_sub-06.nii
```
with this filepath-pattern applied to all 685 filenames and CAT12 output modalities.

In `2_prep_segment.py`, a rerun of CAT12 on the `img_05mm` files should result in e.g.
```
data/img_05mm/CAT12.8.2/mri/p00009_sub-06.nii
```

## Training on simulated data (optional)

[mri_simulate](https://github.com/ChristianGaser/T1-MRI-Phantom) can produce the
training pairs directly. Run it with `simu.affine = 1` and `simu.clean = 1`: it
then writes the T1w and its ground truth on the same 336x384x336 grid of 0.5mm
voxels that `2_prep_segment.py` produces, so no CAT12 run, no affine and no bias
correction are needed on this path.

```matlab
simu = struct('name','sub-01_T1w.nii', 'snrWM',40, 'affine',1, 'clean',1);
mri_simulate(simu, struct('percent',0));
```

Point `simu_path` in `2b_prep_simulate.py` at the derivatives folder and run it.
It pairs every T1w with the label of its anatomy - runs that differ only in
noise, bias field or motion share one label and each becomes its own training
sample - skull-strips, normalizes, derives the 0.75mm versions and writes
`data/csvs/simulated.csv`.

Then set at the top of the training scripts:

```python
csv_name = 'simulated.csv'
eval_suffix = ''   # 3_train_segment.py only
```

`eval_suffix` exists because a simulation has no separate non-bias-corrected
variant, unless you simulate one with `rf.percent` and let it share the label.

Notes:

- The GM `_probseg` that `simu.clean` writes gives an **exact** nogm target,
  the voxels where the triangular decomposition of the label overestimates GM.
  `2_prep_segment.py` can only estimate that from the CAT12 `p1`.
- The fold follows the source subject, so variants of one brain never land on
  both sides of the cross validation.
- `5_train_nogm.py` still reads only the first 5 rows (`[:5]`); drop that if you
  want to use the whole set.
- Motion artefacts belong in the simulation, not in the augmentation: they are a
  global k-space operation and `4_train_segment_patches.py` augments 128^3
  patches, which cannot carry them. Simulate motion variants instead, they share
  the label file.
- `6_train_warp.py` is not covered here, it needs the 1.5mm `p/` inputs from the
  CAT12 path.

### Brain extraction (optional)

`8_train_bet.py` trains the two deepbet models on the same simulations. It needs
a **native space** run (`simu.affine = 0`), because deepbet works on the whole
head: the 168x192x168mm field of view of the registered grid slices through the
skull, and on one test volume 66% of its inferior face and 27% of its anterior
face are still head.

The script reproduces both deepbet stages - the bbox model on the whole volume
at 128^3, the main model on the bounding box of the brain plus a 10% margin at
256^3 - and uses deepbet's own normalization and its own `get_bbox`, so the crop
a model is trained on is the crop it gets at inference. `7_compile_models.py`
traces both with a softmax, because deepbet thresholds the model output directly
while deepmriprep applies the softmax itself for nogm.

The mask is `dseg > 0`, i.e. CAT12's APRG mask, so the result reproduces that and
not the mask deepbet ships. That is the reason to do it: `2_prep_segment.py`
strips with `p0 > 0` while inference strips with deepbet, and training the
extractor on the same mask the segmentation was trained behind removes that
mismatch. Out of domain it will be less robust than the shipped model, which saw
far more heterogeneous data.

## Run scripts
Run the 7 scripts (+read the comments) `1_prep_warp.py`-`7_compile_models.py`!
 
The trained models (e.g. the warp model) can be directly plugged into deepmriprep like this:
```python
from deepmriprep import run_preprocess

run_preprocess(bids_dir='path/to/bids', warp_model_path='path/to/warp_model.pt')
```
