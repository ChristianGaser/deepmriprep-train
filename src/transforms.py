from fastai.basics import np, torch, store_attr, TensorBase, DisplayedTransform
from fastai.vision.all import RandTransform
from niftiai import TensorImage3d, TensorMask3d
from src.synth import tissue_weights, fit_anchors, sample_anchors, shift_contrast


class FlipSagittal(DisplayedTransform):
    order = 10

    def __init__(self, **kwargs):
        store_attr()
        super().__init__(**kwargs)

    def encodes(self, x: (TensorMask3d, TensorImage3d)):
        return x.flip(-3) if x.slices[-3].start + x.slices[-3].stop > 336 else x


class StoreZeroMask(RandTransform):
    order = 45
    def __init__(self, p: float = 1.):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self._zero_mask = TensorBase(b[0] <= 0)

    def encodes(self, x: TensorImage3d):
        x._zero_mask = self._zero_mask
        return x


class ContrastShift(RandTransform):
    """Randomize the tissue contrast of a real image using its p0 label.

    The image is not replaced by a rendering of the label. Only the fitted
    tissue levels are moved, so noise, vessels, dura and every other structure
    that p0 does not describe survive untouched. The input therefore stays
    ambiguous: it is not a function of the target, which a synthesized image
    would be.

    Set p_free > 0 to also draw non-T1w contrasts (WM darkest, etc.). Note that
    the residual keeps its sign, so a voxel brighter than GM stays brighter than
    the new GM level even when the contrast is inverted.

    The residual is deliberately not rescaled with the new contrast span: the
    noise augmentations already randomize SNR on their own.
    """
    order = 46  # after StoreZeroMask(45), so the artefacts act on the new contrast

    def __init__(self, max_shift: float = .25, min_sep: float = .15, p_free: float = .0,
                 ridge: float = 1e-3, p: float = .5):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        if not self.do:
            return
        x, p0 = b[0], b[-1]
        p0 = p0[:, None] if p0.ndim == x.ndim - 1 else p0
        self._w = tissue_weights(TensorBase(p0).float())
        m = fit_anchors(TensorBase(x).float(), self._w, self.ridge)
        self._dm = sample_anchors(m, self.max_shift, self.min_sep, self.p_free) - m

    def encodes(self, x: TensorImage3d):
        return shift_contrast(x, self._w, self._dm)


class BetNormalize(RandTransform):
    """The intensity normalization deepbet applies before its models.

    deepbet does percentile clip -> z-score -> `.226 * x + .449` (the grayscale
    ImageNet statistics) in one step, on the volume it is about to segment. The
    clip needs the percentiles of the whole head and is therefore done once when
    the volumes are prepared; this transform is the rest of it, applied after
    the augmentation so the network sees the same distribution it will see at
    inference. Statistics are per sample, as they are in deepbet.
    """
    order = 90
    split_idx = None  # also on the validation set, it is not an augmentation

    def __init__(self, mean: float = .449, std: float = .226, p: float = 1.):
        super().__init__(p=p)
        store_attr()

    def encodes(self, x: TensorImage3d):
        dims = tuple(range(-3, 0))
        x = x.clamp(min=0, max=1)
        x = (x - x.mean(dims, keepdim=True)) / x.std(dims, keepdim=True).clamp(min=1e-6)
        return self.std * x + self.mean


class ApplyZeroMask(RandTransform):
    order = 82
    def __init__(self, p: float = 1.):
        super().__init__(p=p)
        store_attr()

    def encodes(self, x: TensorImage3d):
        x[x._zero_mask] = 0
        return x


class ScaleIntensity(RandTransform):
    order = 83
    def __init__(self, low: float = .5, high: float = 99.5, p: float = 1.):
        super().__init__(p=p)
        store_attr()

    def encodes(self, x: TensorImage3d):
        x_nonzero = x[~x._zero_mask].cpu()
        x._zero_mask = None
        low = np.percentile(x_nonzero, self.low)
        high = np.percentile(x_nonzero, self.high)
        x = (x - low) / (high - low)
        x[x > 1] = 1 + torch.log10(x[x > 1])
        return x
