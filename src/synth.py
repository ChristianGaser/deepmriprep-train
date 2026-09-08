import torch


def tissue_weights(p0, n_classes=4):
    """Triangular decomposition of a p0 label into tissue fractions.

    Same kernel as `one_hot` in 2_prep_segment.py, but without the background
    class: it contributes no intensity and its weight is zero wherever p0 is 0,
    which keeps a skull-stripped image untouched outside the brain.

    Returns (B, 3, ...) weights for CSF, GM, WM (label values 1, 2, 3).
    """
    return torch.cat([(1 - (p0 - c).abs()).clamp(min=0) for c in range(1, n_classes)], dim=1)


def fit_anchors(x, w, ridge=1e-3):
    """Least squares tissue intensities of x under the label decomposition w.

    Solves min_m ||x - sum_k m_k w_k||^2, i.e. the best piecewise linear image
    that p0 alone can explain. Everything x has beyond it - noise, vessels,
    iron, myelin gradients - is the residual that `shift_contrast` keeps.

    Using all voxels instead of the medians of the pure tissue cores makes this
    work on a patch that holds only a part of a tissue: `ridge` then pulls that
    anchor towards zero, and its weight in the patch is small anyway, so the
    applied change stays small as well.

    Returns (B, 3).
    """
    wf = w.flatten(2)
    xf = x.flatten(2)
    gram = wf @ wf.transpose(1, 2)
    rhs = wf @ xf.transpose(1, 2)
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    lam = ridge * gram.diagonal(dim1=1, dim2=2).mean(-1)
    return torch.linalg.solve(gram + lam[:, None, None] * eye, rhs)[..., 0]


def sample_anchors(m, max_shift=.25, min_sep=.15, p_free=.0):
    """Draw target tissue intensities for a contrast change.

    Shifts are expressed as fractions of the CSF-WM span, so the transform does
    not depend on how the images were normalized.

    With probability `p_free` the three anchors are drawn independently and
    permuted, which gives a non-T1w contrast (WM darkest for a T2w-like image).
    Otherwise each anchor is jittered by up to `max_shift` and the CSF < GM < WM
    order is kept. `min_sep` is the floor on the separation of two anchors:
    without it a draw can collapse the contrast between two tissues and make the
    label unlearnable from the image.

    Returns (B, 3), non-negative.
    """
    b, device = m.shape[0], m.device
    span = (m[:, 2] - m[:, 0]).abs().clamp(min=1e-3)
    sep = min_sep * span

    d = (2 * torch.rand(b, 3, device=device) - 1) * max_shift * span[:, None]
    mono = _separate(m + d, sep)

    lo = (m.min(1).values - .25 * span).clamp(min=0)
    hi = m.max(1).values + .25 * span
    u = torch.sort(torch.rand(b, 3, device=device), dim=1).values
    free = _separate(lo[:, None] + (hi - lo)[:, None] * u, sep)
    free = torch.gather(free, 1, torch.argsort(torch.rand(b, 3, device=device), dim=1))

    m_new = torch.where((torch.rand(b, device=device) < p_free)[:, None], free, mono)
    # a common offset is irrelevant after ScaleIntensity, so the triple is
    # lifted instead of clipped, which would collapse a tissue onto zero
    return m_new - m_new.min(1, keepdim=True).values.clamp(max=0)


def shift_contrast(x, w, dm):
    """Add a label driven intensity shift to a real image.

    x + sum_k dm_k w_k moves each tissue to a new level while the residual of
    the fit stays exactly as it was, so the image never becomes a function of
    the label. Outside the brain all w_k are zero and x is unchanged.
    """
    return x + (dm[(...,) + (None,) * (x.ndim - 2)] * w).sum(1, keepdim=True)


def _separate(v, sep):
    """Push a sorted-ish triple apart so neighbours keep a minimum distance."""
    a = v[:, 0]
    b = torch.maximum(v[:, 1], a + sep)
    c = torch.maximum(v[:, 2], b + sep)
    return torch.stack([a, b, c], dim=1)
