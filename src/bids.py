"""Pairing of mri_simulate outputs by their BIDS entities."""
import re
from pathlib import Path
import pandas as pd


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


def pair_simulations(simu_dir, n_folds=5):
    """Pair every simulated T1w with the label that belongs to its anatomy.

    A label desc holds the anatomical tags plus `Clean`, an image desc holds the
    acquisition tags followed by the same anatomical tags. The label whose
    anatomical part is the longest suffix of the image desc is therefore the
    right one, which keeps a `Wmh2` image away from the label of a run without
    WMHs. A cleaned label wins over an uncleaned one of the same anatomy.

    The fold follows the source subject, otherwise variants of one brain end up
    on both sides of the cross validation.
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
    subjects = sorted(df.subject.unique())
    fold = {s: i % n_folds for i, s in enumerate(subjects)}
    df['fold'] = df.subject.map(fold)
    return df
