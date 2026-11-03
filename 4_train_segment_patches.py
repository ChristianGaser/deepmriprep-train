from fastai.basics import pd, mae, torch, set_seed, Learner
from fastai.data.all import ColReader, ColSplitter, DataBlock, DataLoaders
from niftiai import aug_transforms3d, TensorImage3d
from niftiai.data import ImageBlock3d, MaskBlock3d
from src.augment import Blur3d, ScaledChiNoise3d, Resolution3d
from src.models import Unet3d, StepActivation, TwoInputsUnet3d
from src.transforms import FlipSagittal, StoreZeroMask, ApplyZeroMask, ScaleIntensity, ContrastShift
path = '.'  # if data_path is absolute(=starts with "/") set path = '/'
data_path = 'data'
csv_name = 'openneuro_hd.csv'  # 'simulated.csv' for the mri_simulate outputs


def get_dls(df, img_col, pred_mask_col, mask_col, valid_col, batch_tfms=None, bs=1, item_tfms=None):
    dblock = DataBlock(blocks=[ImageBlock3d(), MaskBlock3d(), MaskBlock3d()],
                       get_x=(ColReader(img_col), ColReader(pred_mask_col)),
                       get_y=ColReader(mask_col),
                       splitter=ColSplitter(valid_col),
                       n_inp=2,
                       batch_tfms=batch_tfms,
                       item_tfms=item_tfms)
    return DataLoaders.from_dblock(dblock, df, path=path, bs=bs)


def patch_string(patch, size):
    string = '['
    for p, s in zip(patch, size):
        string += f'{p}:{p + s},'
    return string[:-1] + ']'


def get_patch_df(df, patch_strings, cols=('img', 'pred_mask', 'mask')):
    dfs = []
    for s in patch_strings:
        sdf = df.copy()
        for c in cols:
            sdf[c] = sdf[c] + s
        dfs.append(sdf)
    return pd.concat(dfs).reset_index(drop=True)


if __name__ == '__main__':
    set_seed(1)
    df = pd.read_csv(f'{data_path}/csvs/{csv_name}')
    df['img'] = f'{data_path}/img_05mm_minmax/' + df.filename + '.nii.gz'
    df['mask'] = f'{data_path}/p0_05mm/' + df.filename + '.nii.gz'
    header = TensorImage3d.create(df.img[0]).header
    patch_size = [128, 128, 128]
    patches = pd.read_csv(f'{data_path}/csvs/patches.csv').values
    patch_strings = [patch_string(p, patch_size) for p in patches]
    # train on full openneuro-hd dataset
    df['pred_mask'] = f'{data_path}/p0_05mm_pred/' + df.filename + '.nii.gz'
    for i in range(-1, 18):
        if i == -1:
            p_strings = patch_strings
        else:
            p_strings = [patch_strings[i], patch_strings[i + 18]] if i in list(range(9)) else patch_strings[i:i + 1]
        patch_df = get_patch_df(df, p_strings)
        patch_df.loc[len(patch_df)] = patch_df.loc[0]
        patch_df['is_valid'] = (len(patch_df) - 2) * [0] + [1, 1]
        batch_tfms = aug_transforms3d(max_warp=0, max_zoom=0, max_rotate=0, max_shear=0, max_translate=3 * .02, p_affine=.2,
                                      max_ghost=.5, max_spike=2., max_bias=.2, max_motion=.5, max_noise=.0, max_down=0,
                                      max_ring=1., max_contrast=.1, max_dof_noise=3, image_mode='nearest',
                                      dims_ghost=(0, 1, 2), n_ghosts=2, p_spike=.1, freq_spike=.5, dims_ring=(0, 1, 2),
                                      max_move=3 * .02, p_flip=.5 if i >= 9 and len(p_strings) < 5 else .0)
        # p_free>0 also draws non-T1w contrasts (e.g. WM darkest); set it to 0 to stay T1w only
        # max_down=0: mriaug reduces and interpolates back with nearest on one fixed
        # axis, Resolution3d replaces it with an area average and a spline
        batch_tfms += [StoreZeroMask(), ContrastShift(max_shift=.25, p_free=.25, p=.5),
                       Resolution3d(max_scale=3., p_iso=.3, p=.15),
                       ScaledChiNoise3d(.1, p=.1), Blur3d(.5, p=.1), ApplyZeroMask(), ScaleIntensity()]
        dls = get_dls(patch_df, img_col='img', pred_mask_col='pred_mask', mask_col='mask', valid_col='is_valid',
                      bs=2, batch_tfms=batch_tfms, item_tfms=[FlipSagittal()] if 0 <= i < 9 else None)
        model = TwoInputsUnet3d(Unet3d(n_in=2, n_out=1, n_ch=8), StepActivation())
        learn = Learner(dls, model=model, loss_func=mae)
        if i != -1:
            learn.model.load_state_dict(torch.load(f'{data_path}/models/segmentation_model_patch_-1.pth'))
        learn.fit_one_cycle(20 if i >= 9 else 10 if i != -1 else 2, 1e-3)
        torch.save(learn.model.state_dict(), f'{data_path}/models/segmentation_model_patch_{i}.pth')
    # train cross validation
    for fold in range(5):
        set_seed(1)
        df['is_valid'] = df.fold == fold
        df['pred_mask'] = f'{data_path}/p0_05mm_pred/' + df.filename + f'_fold{fold}.nii.gz'
        for i in range(-1, 18):
            if i == -1:
                p_strings = patch_strings
            else:
                p_strings = [patch_strings[i], patch_strings[i + 18]] if i in list(range(9)) else patch_strings[i:i + 1]
            patch_df = get_patch_df(df, p_strings)
            batch_tfms = aug_transforms3d(max_warp=0, max_zoom=0, max_rotate=0, max_shear=0, max_translate=3 * .02, p_affine=.2,
                                          max_ghost=.5, max_spike=2., max_bias=.2, max_motion=.5, max_noise=.0, max_down=0,
                                          max_ring=1., max_contrast=.1, max_dof_noise=3, mode='nearest',
                                          dims_ghost=(0, 1, 2), n_ghosts=2, p_spike=.1, freq_spike=.5, dims_ring=(0, 1, 2),
                                          max_move=3 * .02, p_flip=.5 if i >= 9 and len(p_strings) < 5 else .0)
            batch_tfms += [StoreZeroMask(), ContrastShift(max_shift=.25, p_free=.25, p=.5),
                           Resolution3d(max_scale=3., p_iso=.3, p=.15),
                           ApplyZeroMask(), ScaleIntensity()]
            dls = get_dls(patch_df, img_col='img', pred_mask_col='pred_mask', mask_col='mask', valid_col='is_valid',
                          bs=2, batch_tfms=batch_tfms, item_tfms=[FlipSagittal()] if 0 <= i < 9 else None)
            model = TwoInputsUnet3d(Unet3d(n_in=2, n_out=1, n_ch=8), StepActivation())
            learn = Learner(dls, model=model, loss_func=mae)
            if i != -1:
                learn.model.load_state_dict(torch.load(f'{data_path}/models/segmentation_model_patch_-1_fold{fold}.pth'))
            learn.fit_one_cycle(20 if i >= 9 else 10 if i != -1 else 2, 1e-3)
            torch.save(learn.model.state_dict(), f'{data_path}/models/segmentation_model_patch_-1_fold{fold}.pth')
