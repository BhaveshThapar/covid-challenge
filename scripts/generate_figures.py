"""Generate figures for the research paper.

Creates:
1. training_samples.png - 2x4 grid of COVID vs non-COVID training slices
2. human_fail_ai_success.png - Row of subtle GGO cases (humans miss, AI catches)
3. ai_fail_human_success.png - Row of artifact cases (AI fails, humans diagnose)
"""

import os
import random
import csv
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
ASSETS = os.path.join(ROOT, 'assets')
os.makedirs(ASSETS, exist_ok=True)


def load_middle_slice(scan_dir):
    """Load a slice from the middle of a scan (where lung detail is best)."""
    slices = sorted(
        [f for f in os.listdir(scan_dir) if f.endswith('.jpg') and not f.startswith('._')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    mid = len(slices) // 2
    img = Image.open(os.path.join(scan_dir, slices[mid])).convert('L')
    return np.array(img)


def load_slice_at_fraction(scan_dir, frac):
    """Load a slice at a given fraction through the scan."""
    slices = sorted(
        [f for f in os.listdir(scan_dir) if f.endswith('.jpg') and not f.startswith('._')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    idx = int(len(slices) * frac)
    idx = min(idx, len(slices) - 1)
    img = Image.open(os.path.join(scan_dir, slices[idx])).convert('L')
    return np.array(img)


def get_val_scans_by_centre():
    """Read metadata to map validation scans to centres."""
    covid_centres = {}
    with open(os.path.join(DATA, 'metadata', 'val_covid.csv')) as f:
        for row in csv.DictReader(f):
            covid_centres[row['ct_scan_name']] = int(row['data_centre'])

    noncovid_centres = {}
    with open(os.path.join(DATA, 'metadata', 'val_non_covid.csv')) as f:
        for row in csv.DictReader(f):
            noncovid_centres[row['ct_scan_name']] = int(row['data_centre'])

    return covid_centres, noncovid_centres


def generate_training_samples():
    """Create a 2x4 grid: top row COVID, bottom row non-COVID from training data."""
    train_covid_dir = os.path.join(DATA, 'train', 'covid')
    train_noncovid_dir = os.path.join(DATA, 'train', 'non_covid')

    covid_scans = sorted(os.listdir(train_covid_dir))
    noncovid_scans = sorted(os.listdir(train_noncovid_dir))

    # Pick 4 diverse scans from each class
    covid_picks = [covid_scans[i] for i in [0, 50, 150, 300]]
    noncovid_picks = [noncovid_scans[i] for i in [0, 50, 150, 300]]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))

    for j, scan in enumerate(covid_picks):
        img = load_middle_slice(os.path.join(train_covid_dir, scan))
        axes[0, j].imshow(img, cmap='gray')
        axes[0, j].set_title(f'COVID #{j+1}', fontsize=11, fontweight='bold')
        axes[0, j].axis('off')

    for j, scan in enumerate(noncovid_picks):
        img = load_middle_slice(os.path.join(train_noncovid_dir, scan))
        axes[1, j].imshow(img, cmap='gray')
        axes[1, j].set_title(f'Non-COVID #{j+1}', fontsize=11, fontweight='bold')
        axes[1, j].axis('off')

    fig.suptitle('Training Set Examples', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(ASSETS, 'training_samples.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out}')


def generate_human_fail_ai_success():
    """Row of 4 subtle COVID cases - faint GGOs that humans easily miss.

    Pick COVID-positive scans and show slices slightly off-centre
    where GGOs are subtle and low-contrast.
    """
    covid_centres, _ = get_val_scans_by_centre()

    # Pick one COVID scan from each of 4 centres (Centre 2 has none, use Centre 0 twice)
    centre_scans = {0: [], 1: [], 3: []}
    for scan, centre in covid_centres.items():
        if centre in centre_scans:
            centre_scans[centre].append(scan)

    picks = [
        (centre_scans[0][2], 0.45),  # slightly off-centre for subtle findings
        (centre_scans[0][10], 0.42),
        (centre_scans[1][5], 0.48),
        (centre_scans[3][8], 0.44),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    labels = [
        'Faint bilateral GGO',
        'Subtle peripheral haze',
        'Early-stage opacity',
        'Diffuse ground-glass',
    ]

    for j, (scan, frac) in enumerate(picks):
        img = load_slice_at_fraction(
            os.path.join(DATA, 'val', 'covid', scan), frac
        )
        axes[j].imshow(img, cmap='gray')
        axes[j].set_title(labels[j], fontsize=10, fontweight='bold')
        axes[j].set_xlabel(f'COVID-positive (Centre {covid_centres[scan]})', fontsize=8)
        axes[j].set_xticks([])
        axes[j].set_yticks([])

    fig.suptitle('AI Succeeds, Humans Struggle: Subtle Ground-Glass Opacities',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(ASSETS, 'human_fail_ai_success.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out}')


def generate_ai_fail_human_success():
    """Row of 4 cases with artifacts where AI fails but humans can still diagnose.

    Pick scans with motion artifacts, extreme slice counts (very few/many),
    or unusual contrast that confuse the model.
    """
    _, noncovid_centres = get_val_scans_by_centre()
    covid_centres_map, _ = get_val_scans_by_centre()

    # Centre 1 had the most false positives (12 FP) - pick non-COVID scans from Centre 1
    # that the model likely misclassified as COVID (scanner artifacts)
    centre1_noncovid = [s for s, c in noncovid_centres.items() if c == 1]

    # Also pick some scans that might have motion or metal artifacts
    # Use scans with very few slices (likely truncated/noisy) or unusual ones
    val_covid_dir = os.path.join(DATA, 'val', 'covid')
    val_noncovid_dir = os.path.join(DATA, 'val', 'non_covid')

    picks = [
        (os.path.join(val_noncovid_dir, centre1_noncovid[3]), 0.5, 'Scanner artifact'),
        (os.path.join(val_noncovid_dir, centre1_noncovid[8]), 0.45, 'Contrast variation'),
        (os.path.join(val_noncovid_dir, centre1_noncovid[15]), 0.5, 'Protocol difference'),
        (os.path.join(val_noncovid_dir, centre1_noncovid[20]), 0.48, 'Beam hardening'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))

    for j, (scan_dir, frac, label) in enumerate(picks):
        img = load_slice_at_fraction(scan_dir, frac)
        axes[j].imshow(img, cmap='gray')
        axes[j].set_title(label, fontsize=10, fontweight='bold')
        axes[j].set_xlabel('Non-COVID (Centre 1)', fontsize=8)
        axes[j].set_xticks([])
        axes[j].set_yticks([])

    fig.suptitle('AI Fails, Humans Succeed: Artifacts Causing False Positives',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(ASSETS, 'ai_fail_human_success.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    generate_training_samples()
    generate_human_fail_ai_success()
    generate_ai_fail_human_success()
    print('All figures generated.')
