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

## Run scripts
Run the 7 scripts (+read the comments) `1_prep_warp.py`-`7_compile_models.py`!
 
The trained models (e.g. the warp model) can be directly plugged into deepmriprep like this:
```python
from deepmriprep import run_preprocess

run_preprocess(bids_dir='path/to/bids', warp_model_path='path/to/warp_model.pt')
```
