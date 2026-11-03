from tqdm import tqdm
from fastai.basics import np, pd, mae, torch, set_seed, Path, Learner
from niftiai import aug_transforms3d, TensorImage3d, SegmentationDataLoaders3d
from spline_resize import resize
from src.augment import Blur3d, ScaledChiNoise3d, Resolution3d
from src.models import Unet3d, StepActivation
from src.transforms import StoreZeroMask, ApplyZeroMask, ScaleIntensity, ContrastShift
set_seed(1)
path = '.'  # if data_path is absolute(=starts with "/") set path = '/'
data_path = 'data'
csv_name = 'openneuro_hd.csv'  # 'simulated.csv' for the mri_simulate outputs
eval_suffix = '_raw'  # the uncorrected images used at prediction time; '' for simulated data,
                      # which has no separate non-bias-corrected variant
Path(f'{data_path}/models').mkdir(exist_ok=True)
Path(f'{data_path}/p0_05mm_pred').mkdir(exist_ok=True)
nib_affine_05mm = np.array([[.5, 0, 0, -84], [0, .5, 0, -120], [0, 0, .5, -72], [0, 0, 0, 0]])

shape = (336, 384, 336)
df = pd.read_csv(f'{data_path}/csvs/{csv_name}')
df['img'] = f'{data_path}/img_075mm_minmax/' + df.filename + '.nii.gz'
df['mask'] = f'{data_path}/p0_075mm/' + df.filename + '.nii.gz'
header = TensorImage3d.create(df['mask'].iloc[0]).header
batch_tfms = aug_transforms3d(max_warp=0, max_zoom=0, max_rotate=0, max_shear=0, max_translate=.02, p_affine=.2,
                              max_ghost=.5, max_spike=2., max_bias=.2, max_motion=.5, max_noise=.0, max_down=0,
                              max_ring=1., max_contrast=.1, max_dof_noise=3, image_mode='nearest',
                              dims_ghost=(0, 1, 2), n_ghosts=2, p_spike=.1, freq_spike=.5, dims_ring=(0, 1, 2))
# p_free>0 also draws non-T1w contrasts (e.g. WM darkest); set it to 0 to stay T1w only
# max_down=0: mriaug reduces and interpolates back with nearest on one fixed
# axis, Resolution3d replaces it with an area average and a spline
batch_tfms += [StoreZeroMask(), ContrastShift(max_shift=.25, p_free=.25, p=.5),
               Resolution3d(max_scale=3., p_iso=.3, p=.15),
               ScaledChiNoise3d(.1, p=.1), Blur3d(.5, p=.1), ApplyZeroMask(), ScaleIntensity()]
model = torch.nn.Sequential(Unet3d(n_out=1), StepActivation())
# train on full openneuro-hd dataset
df_total = df.copy()
df_total.loc[len(df_total)] = df_total.loc[0]
df_total['is_valid'] = (len(df_total) - 1) * [0] + [1]
dls = SegmentationDataLoaders3d.from_df(df_total, path=path, fn_col='img', label_col='mask',
                                        valid_col='is_valid', bs=1, batch_tfms=batch_tfms)
learn = Learner(dls, model=model, loss_func=mae)
learn.model = learn.model.cuda()
learn.fit_one_cycle(60, 1e-3)
torch.save(learn.model.state_dict(), f'{data_path}/models/segmentation_model.pth')
#learn.model.load_state_dict(torch.load(f'{DATA_PATH}/models/segmentation_model.pth')) # load model
for fp in tqdm(df_total.img[:-1]):
    filename = Path(fp).stem.split('.')[0]
    x = TensorImage3d.create(fp.replace('_minmax', '_minmax' + eval_suffix)).cuda()
    with torch.no_grad():
        p = learn.model(x[None])
        p = resize(p, shape, align_corners=True, mask_value=0)[0]
    p = TensorImage3d(p, affine=nib_affine_05mm, header=header)
    p.header.set_data_dtype(np.uint8)
    p.save(f'{data_path}/p0_05mm_pred/{filename}.nii.gz')
# train cross validation
for fold in range(5):
    set_seed(1)
    df['is_valid'] = df.fold == fold
    dls = SegmentationDataLoaders3d.from_df(df, path=path, fn_col='img', label_col='mask',
                                            valid_col='is_valid', bs=1, batch_tfms=batch_tfms)
    learn = Learner(dls, model=model, loss_func=mae)
    learn.model = learn.model.cuda()
    learn.fit_one_cycle(60, 1e-3)
    torch.save(learn.model.state_dict(), f'{data_path}/model/segmentation_model_fold{fold}.pth')
    # learn.model.load_state_dict(torch.load(f'{DATA_PATH}/models/segmentation_model_fold{fold}.pth')) # load model
    for fp in tqdm(df.img):
        filename = Path(fp).stem.split('.')[0]
        x = TensorImage3d.create(fp.replace('_minmax', '_minmax' + eval_suffix)).cuda()
        with torch.no_grad():
            p = learn.model(x[None])
            p = resize(p, shape, align_corners=True, prefilter=True, mask_value=0)[0]
        p = TensorImage3d(p, affine=nib_affine_05mm, header=header)
        p.header.set_data_dtype(np.uint8)
        p.save(f'{data_path}/p0_05mm_pred/{filename}_fold{fold}.nii.gz')
