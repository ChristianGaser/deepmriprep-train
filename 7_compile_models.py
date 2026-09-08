import glob
import torch
from tqdm import tqdm
from src.models import SyMNet, Unet3d, StepActivation, TwoInputsUnet3d
data_path = 'data'


for fp in tqdm(sorted(glob.glob(f'{data_path}/models/*.pth'))):
    print(fp)
    wrap = None
    if 'warp' in fp:
        shape = [4, 113, 137, 113]
        model = SyMNet()
    elif 'bbox' in fp:  # must be tested before brain_extraction, the name holds both
        shape = [1, 128, 128, 128]
        model = Unet3d(n_in=1, n_out=2, n_ch=16)
        wrap = torch.nn.Softmax(dim=1)
    elif 'brain_extraction' in fp:
        shape = [1, 256, 256, 256]
        model = Unet3d(n_in=1, n_out=2, n_ch=8)
        wrap = torch.nn.Softmax(dim=1)
    elif 'nogm' in fp:
        shape = [1, 128, 288, 256]
        model = Unet3d(n_in=1, n_out=2, n_ch=8)
    elif 'patch' in fp:
        shape = [2, 128, 128, 128]
        model = TwoInputsUnet3d(Unet3d(n_in=2, n_out=1, n_ch=8), StepActivation())
    else:
        shape = [1, 224, 256, 224]
        model = torch.nn.Sequential(Unet3d(n_in=1, n_out=1, n_ch=8), StepActivation())
    model.load_state_dict(torch.load(fp))
    # deepbet thresholds the model output directly, so the softmax has to be part
    # of the traced graph. deepmriprep applies it itself for nogm, which is why
    # that one stays bare.
    if wrap is not None:
        model = torch.nn.Sequential(model, wrap)
    model.eval()
    model = torch.jit.trace(model, torch.rand(1, *shape), strict=True)
    torch.jit.save(model, fp[:-1])
