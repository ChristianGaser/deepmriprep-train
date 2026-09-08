from fastai.basics import pd, torch, set_seed, Path, Learner
from niftiai import aug_transforms3d, SegmentationDataLoaders3d
from src.loss import DiceFocalLoss
from src.models import Unet3d
path = '.'  # if data_path is absolute(=starts with "/") set path = '/'
data_path = 'data'
csv_name = 'openneuro_hd.csv'  # 'simulated.csv' gives exact nogm targets from the GM probseg
Path(f'{data_path}/models').mkdir(exist_ok=True)


def get_patch_df(df, patch_strings, cols=('img', 'mask')):
    dfs = []
    for s in patch_strings:
        sdf = df.copy()
        for c in cols:
            sdf[c] = sdf[c] + s
        dfs.append(sdf)
    return pd.concat(dfs).reset_index(drop=True)


if __name__ == '__main__':
    set_seed(1)
    df = pd.read_csv(f'{data_path}/csvs/{csv_name}')[:5]  # NOTE: only the first 5 subjects
    df['img'] = f'{data_path}/p0_05mm/' + df.filename + '.nii.gz'
    df['mask'] = f'{data_path}/nogm/' + df.filename + '.nii.gz'
    size = (128, 288, 256)
    p_strings = ['[56:184,28:316,0:256]', '[152:280,28:316,0:256]']
    batch_tfms = aug_transforms3d(max_translate=.02, p_affine=.2, max_warp=0, max_zoom=0, max_rotate=0, max_shear=0,
                                  max_ghost=0, max_spike=0, max_bias=0, max_motion=0, max_noise=0, max_down=0,
                                  max_ring=0, max_contrast=0, p_flip=0, image_mode='nearest')
    loss_func = DiceFocalLoss(lambda_focal=1., lambda_gdl=1., cls_props=[.99, .01], include_background=False)
    # train on full openneuro-hd dataset
    df_total = df.copy()
    df_total = get_patch_df(df_total, p_strings)
    df_total.loc[len(df_total)] = df_total.loc[0]
    df_total['is_valid'] = (len(df_total) - 1) * [0] + [1]
    dls = SegmentationDataLoaders3d.from_df(df_total, path=path, fn_col='img', label_col='mask',
                                            valid_col='is_valid', bs=1, batch_tfms=batch_tfms)
    learn = Learner(dls, model=Unet3d(n_out=2, n_ch=8), loss_func=loss_func)
    learn.fit_one_cycle(30, 1e-3)
    torch.save(learn.model.state_dict(), f'{data_path}/models/nogm_model.pth')
    # train cross validation
    df = get_patch_df(df, p_strings)
    for fold in range(5):
        set_seed(1)
        df['is_valid'] = df.fold == fold
        dls = SegmentationDataLoaders3d.from_df(df, path=path, fn_col='img', label_col='mask',
                                                valid_col='is_valid', bs=1, batch_tfms=batch_tfms)
        learn = Learner(dls, model=Unet3d(n_out=2, n_ch=8), loss_func=loss_func)
        learn.fit_one_cycle(30, 1e-3)
        torch.save(learn.model.state_dict(), f'{data_path}/models/nogm_model_fold{fold}.pth')
