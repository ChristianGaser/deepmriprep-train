from fastai.basics import torch, random, store_attr, F
from fastai.vision.all import RandTransform
from mriaug.utils import to_ndim
from niftiai import TensorImage3d
from spline_resize import resize


class ScaledChiNoise3d(RandTransform):  # to compensate for initial affine transformation
    order = 50
    def __init__(self, max_intensity: float = .1, max_downsample: float = 3., max_dof: int = 3, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        x = b[0]
        dof = random.randint(1, self.max_dof)
        shape = list(x.shape[:-3]) + [min(int(s / (1 + self.max_downsample * random.random())), s) for s in x.shape[-3:]]
        self._noise = to_ndim(self.max_intensity, x.ndim) * torch.randn([*(shape[(1 if self.batch else 0):]), dof], device=x.device)
        # Only works with single channel images!
        self._noise = F.interpolate(self._noise[:, 0].permute(0, 4, 1, 2, 3), x.shape[-3:], mode='trilinear').permute(0, 2, 3, 4, 1)[:, None]

    def encodes(self, x: TensorImage3d):
        return ((x[..., None] + self._noise) ** 2).mean(-1).sqrt()


class Resolution3d(RandTransform):
    """Simulate a lower acquisition resolution: average over the slice profile,
    then interpolate back onto the grid the network works on.

    A thick slice is the integral of the tissue over the slice profile, not a
    point sample, so the reduction has to be an average. `area` is exactly that
    average over the bin, i.e. a rectangular slice profile. mriaug's
    Downsample3d reduces with `nearest`, which keeps one sample per bin and
    discards the rest, so it aliases instead of producing partial volume, and it
    interpolates back with `nearest` as well, which no reconstruction does.
    Measured on a 0.5mm simulation reduced to 1.5mm slices and back, against the
    undegraded volume inside the head: nearest/nearest gives an rmse of 48.8,
    area/spline 31.8.

    The way back is spline_resize.resize, the same interpolation that
    2_prep_segment.py and 3_train_segment.py use between 0.5 and 0.75mm.

    Downsample3d is fixed to a single axis (`dims` defaults to the integer 2),
    so it only ever simulates thick axial slices. This covers a thick slice
    along any of the three axes and, with probability p_iso, an isotropic loss -
    a 1mm scan interpolated onto the 0.5mm grid, which is what most real data
    actually is.

    On a patch this is accurate up to the patch border, where the average and
    the spline would have drawn on voxels outside it. Unlike motion, which is a
    global k-space operation, the error stays local.

    Because the stored images are already skull-stripped, the average blends the
    brain edge towards zero and not towards the CSF and skull that a real thick
    slice would mix in. Measured at 3x along one axis, the brain core keeps its
    intensity to +0.2% while the outer 1.5mm darkens by 9%. The leak into the
    background - 1.75M voxels on a 0.5mm volume - is what makes the order below
    matter.
    """
    order = 48  # after ContrastShift(46) and before ScaledChiNoise3d(50) and
                # Blur3d(52), since a thick slice averages the noise too, and
                # before ApplyZeroMask(82), which re-zeros the background that
                # the average leaked into

    def __init__(self, max_scale: float = 3., dims: tuple = (0, 1, 2), p_iso: float = .3, p: float = .15):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        factor = 1 + (self.max_scale - 1) * random.random()
        if random.random() < self.p_iso:
            self._factors = 3 * [factor]
        else:
            dim = random.choice(self.dims)
            self._factors = [factor if i == dim else 1. for i in range(3)]

    def encodes(self, x: TensorImage3d):
        size = [max(1, round(s / f)) for s, f in zip(x.shape[-3:], self._factors)]
        if list(x.shape[-3:]) == size:
            return x
        low = F.interpolate(x, size=size, mode='area')
        return resize(low, x.shape[-3:], align_corners=True, mask_value=0).to(x.dtype)


class Blur3d(RandTransform):
    order = 52

    def __init__(self, max_sigma=.5, kernel_size=7, p=.1):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        kernel = gaussian_smoothing_kernel(self.kernel_size, random.random() * self.max_sigma)
        self._kernel = kernel[None, None].repeat(b[0].shape[1], 1, 1, 1, 1).to(b[0].device)

    def encodes(self, x: TensorImage3d):
        return F.conv3d(F.pad(x, [self._kernel.shape[-1] // 2] * 6, mode='reflect'), self._kernel, groups=x.shape[1])


def gaussian_smoothing_kernel(kernel_size, sigma, normalize=True):
    kernel_size = 3 * [kernel_size] if isinstance(kernel_size, int) else kernel_size
    sigma = 3 * [sigma] if isinstance(sigma, float) else sigma
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * (2 * torch.pi)**.5) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
    return kernel / kernel.sum() if normalize else kernel


def gauss_smoothing(x, sigma=3., kernel_size=3):
    kernel = gaussian_smoothing_kernel(kernel_size, sigma).to(x.device)
    kernel = kernel[None, None].repeat(x.shape[1], 1, 1, 1, 1)
    x = F.pad(x, [kernel_size // 2] * 6, mode='reflect')
    return F.conv3d(x.type(torch.float32), kernel, groups=x.shape[1]).type(x.dtype)
