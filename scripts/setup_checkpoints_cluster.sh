#!/bin/bash
# Setup checkpoints on the cluster to mirror local structure.
# Run from project root: bash scripts/setup_checkpoints_cluster.sh
#
# Expected layout after setup:
#   checkpoints/
#   ├── v1_ovr_best.pt      # DINOv2 (from your training)
#   ├── v4_ovr_best.pt      # DenseNet (get from Aadit)
#   ├── best.pt             # EfficientNet (downloaded from HuggingFace)
#   └── radimagenet_densenet121.pt  # RadImageNet for DenseNet init
set -e
cd "$(dirname "$0")/.."
mkdir -p checkpoints

echo "=== Checkpoint setup ==="
echo "1. DINOv2 (v1_ovr_best.pt): Place from your training run or scp from local"
echo "2. DenseNet (v4_ovr_best.pt): Get from Aadit"
echo "3. EfficientNet (best.pt): Downloading from HuggingFace..."

if [ ! -f checkpoints/best.pt ]; then
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='BhaveshThapar/covid-checkpoints', filename='best.pt', local_dir='checkpoints')
print('Downloaded best.pt')
" 2>/dev/null || {
    echo "  Install: pip install huggingface_hub"
    echo "  Or: huggingface-cli download BhaveshThapar/covid-checkpoints best.pt --local-dir checkpoints"
}
else
    echo "  best.pt exists"
fi

echo ""
echo "Checkpoint status:"
for f in checkpoints/v1_ovr_best.pt checkpoints/v4_ovr_best.pt checkpoints/best.pt checkpoints/radimagenet_densenet121.pt; do
    [ -f "$f" ] && echo "  OK $f" || echo "  -- $f (missing)"
done
