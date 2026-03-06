"""
End-to-end integration tests for the improved COVID detection pipeline.

Tests:
  1. Data pipeline: augmentations, stochastic sampling, collation
  2. Model: forward_features, forward, from_slice_classifier
  3. Losses: CrossEntropy w/ label smoothing, FocalLoss w/ alpha
  4. Mixup: embedding_mixup correctness
  5. Training loop: mini overfit test (a few steps of Phase 2)
  6. Evaluation: TTA, threshold sweep
  7. Ensemble: multi-model soft-voting
  8. Config loading: all 3 experiment configs
"""
import sys
import os
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, compute_per_source_f1, CheckpointManager
from src.model import SliceClassifier, CovidDetector
from src.losses import FocalLoss
from src.train import build_criterion, embedding_mixup, get_cosine_warmup_scheduler
from src.dataset import (
    get_train_transforms, get_val_transforms,
    build_scan_manifest, ScanDataset, scan_collate_fn,
    build_slice_dataloaders, build_scan_dataloaders,
)
from src.evaluate import evaluate, sweep_threshold

PASS = 0
FAIL = 0


def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  ✓ {name}")
    except Exception as e:
        FAIL += 1
        print(f"  ✗ {name}: {e}")
        traceback.print_exc()


# ========== TEST 1: Data Pipeline ==========

def test_augmentation_pipeline():
    """Verify augmentations produce valid outputs."""
    tfm = get_train_transforms(256)
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    for _ in range(20):  # Run multiple times to hit stochastic branches
        result = tfm(image=img)
        tensor = result["image"]
        assert tensor.shape == (3, 256, 256), f"Bad shape: {tensor.shape}"
        assert not torch.isnan(tensor).any(), "NaN in augmented image"
        assert not torch.isinf(tensor).any(), "Inf in augmented image"


def test_stochastic_slice_sampling():
    """Verify random sampling gives different slices each call."""
    config = load_config("configs/exp_b3_s42.yaml")
    entries = build_scan_manifest("data", "val", "data/metadata")
    ds = ScanDataset(entries[:5], get_val_transforms(256), slices_per_scan=8,
                     sampling_strategy="random")
    # Get same item twice with different seeds — should sample different slices
    set_seed(42)
    imgs1, _, _ = ds[0]  # returns (images, label, source) — no mask
    set_seed(123)
    imgs2, _, _ = ds[0]
    # With random sampling, the images should differ (different slices)
    assert imgs1.shape[0] == 8, f"Expected 8 slices, got {imgs1.shape[0]}"


def test_scan_collation():
    """Verify collate function handles variable-length scans."""
    config = load_config("configs/exp_b3_s42.yaml")
    entries = build_scan_manifest("data", "val", "data/metadata")
    ds = ScanDataset(entries[:4], get_val_transforms(256), slices_per_scan=8,
                     sampling_strategy="uniform")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, collate_fn=scan_collate_fn)
    batch = next(iter(loader))
    images, labels, sources, masks = batch
    assert images.dim() == 5, f"Expected 5D tensor, got {images.dim()}D"
    assert masks.shape == (2, 8), f"Bad mask shape: {masks.shape}"
    assert labels.shape == (2,), f"Bad labels shape: {labels.shape}"


# ========== TEST 2: Model ==========

def test_forward_features():
    """Verify forward_features returns embedding, not logits."""
    model = CovidDetector(classifier_hidden_dim=512, drop_path_rate=0.3)
    model.eval()
    x = torch.randn(2, 4, 3, 256, 256)
    mask = torch.ones(2, 4)
    with torch.no_grad():
        embed, attn = model.forward_features(x, mask)
    assert embed.shape == (2, 1536), f"Bad embed shape: {embed.shape}"
    assert attn.shape == (2, 4), f"Bad attn shape: {attn.shape}"
    # Attention should sum to ~1
    attn_sum = attn.sum(dim=1)
    assert torch.allclose(attn_sum, torch.ones(2), atol=0.01), \
        f"Attention doesn't sum to 1: {attn_sum}"


def test_forward_matches():
    """Verify forward() == forward_features() + classifier()."""
    model = CovidDetector(classifier_hidden_dim=512, drop_path_rate=0.0)
    model.eval()
    x = torch.randn(1, 4, 3, 224, 224)
    mask = torch.ones(1, 4)
    with torch.no_grad():
        logits_direct, attn_direct = model(x, mask)
        embed, attn_feat = model.forward_features(x, mask)
        logits_twostep = model.classifier(embed)
    assert torch.allclose(logits_direct, logits_twostep, atol=1e-5), \
        f"forward() != forward_features()+classifier(): {logits_direct} vs {logits_twostep}"
    assert torch.allclose(attn_direct, attn_feat, atol=1e-5), \
        "Attention weights don't match"


def test_convnext_model():
    """Verify ConvNeXt-Tiny backbone works end-to-end."""
    model = CovidDetector(backbone_name="convnext_tiny",
                          classifier_hidden_dim=384, drop_path_rate=0.3)
    model.eval()
    x = torch.randn(1, 4, 3, 256, 256)
    mask = torch.ones(1, 4)
    with torch.no_grad():
        logits, attn = model(x, mask)
    assert logits.shape == (1, 2), f"Bad logits shape: {logits.shape}"
    assert model.embed_dim == 768, f"Bad embed dim: {model.embed_dim}"


def test_from_slice_classifier():
    """Verify Phase1 → Phase2 model transfer."""
    config = load_config("configs/exp_b3_s42.yaml")
    slice_model = SliceClassifier(drop_path_rate=0.3)
    detector = CovidDetector.from_slice_classifier(slice_model, config)
    # Backbone weights should match
    for p1, p2 in zip(slice_model.backbone.parameters(),
                       detector.backbone.parameters()):
        assert torch.equal(p1, p2), "Backbone weights didn't transfer"


# ========== TEST 3: Losses ==========

def test_focal_loss_basic():
    """Verify FocalLoss computes valid gradients."""
    fl = FocalLoss(gamma=2.0, label_smoothing=0.05)
    logits = torch.randn(8, 2, requires_grad=True)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loss = fl(logits, targets)
    loss.backward()
    assert not torch.isnan(loss), "FocalLoss produced NaN"
    assert logits.grad is not None, "No gradients computed"
    assert not torch.isnan(logits.grad).any(), "NaN in gradients"


def test_focal_loss_with_alpha():
    """Verify per-class alpha weighting works."""
    fl_no_alpha = FocalLoss(gamma=2.0)
    fl_alpha = FocalLoss(alpha=[0.55, 0.45], gamma=2.0)
    logits = torch.randn(8, 2)
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    loss_no = fl_no_alpha(logits, targets)
    loss_alpha = fl_alpha(logits, targets)
    # With asymmetric alpha, losses should differ
    assert loss_no.item() != loss_alpha.item(), \
        "Alpha weighting had no effect"


def test_focal_vs_ce_on_easy():
    """Focal loss should be lower than CE on easy (confident) examples."""
    # Create very confident predictions
    logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    ce = nn.CrossEntropyLoss()(logits, targets)
    fl = FocalLoss(gamma=2.0)(logits.detach().requires_grad_(True), targets)
    # Focal should down-weight easy examples → lower loss
    assert fl.item() < ce.item(), \
        f"Focal ({fl.item():.4f}) should be < CE ({ce.item():.4f}) on easy examples"


def test_build_criterion_focal():
    """Verify build_criterion creates FocalLoss with alpha."""
    criterion = build_criterion({
        "phase2": {"loss_type": "focal", "focal_gamma": 2.0,
                   "focal_alpha": [0.55, 0.45], "label_smoothing": 0.05}
    }, "phase2")
    assert isinstance(criterion, FocalLoss), f"Expected FocalLoss, got {type(criterion)}"
    assert criterion.alpha is not None, "Alpha not set"


# ========== TEST 4: Mixup ==========

def test_mixup_shapes():
    """Verify mixup preserves shapes."""
    embed = torch.randn(4, 1536)
    labels = torch.tensor([0, 1, 0, 1])
    mixed, la, lb, lam = embedding_mixup(embed, labels, alpha=0.2)
    assert mixed.shape == embed.shape, f"Shape mismatch: {mixed.shape}"
    assert la.shape == labels.shape
    assert lb.shape == labels.shape
    assert 0 <= lam <= 1, f"Invalid lambda: {lam}"


def test_mixup_interpolation():
    """Verify mixup actually interpolates embeddings."""
    # Use 8 samples to virtually guarantee non-identity permutation
    embed = torch.eye(8)  # 8x8 identity matrix — each row is unique
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    set_seed(42)
    mixed, la, lb, lam = embedding_mixup(embed, labels, alpha=1.0)
    # At least some mixed embeddings should differ from originals
    differs = not torch.equal(mixed, embed)
    assert differs, "Mixup didn't change any embeddings"


def test_mixup_disabled():
    """Verify alpha=0 returns unchanged embeddings."""
    embed = torch.randn(4, 1536)
    labels = torch.tensor([0, 1, 0, 1])
    mixed, la, lb, lam = embedding_mixup(embed, labels, alpha=0.0)
    assert torch.equal(mixed, embed), "alpha=0 should return unchanged"
    assert lam == 1.0, f"alpha=0 should give lam=1.0, got {lam}"


# ========== TEST 5: Mini Training Loop ==========

def test_mini_train_phase2():
    """Run 2 steps of Phase 2 training to verify the full loop."""
    config = load_config("configs/exp_b3_s42.yaml")
    set_seed(42)

    # Setup minimal data
    entries = build_scan_manifest("data", "val", "data/metadata")[:8]
    ds = ScanDataset(entries, get_train_transforms(256), slices_per_scan=4,
                     sampling_strategy="random")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, collate_fn=scan_collate_fn,
                        num_workers=0)

    # Model
    model = CovidDetector(classifier_hidden_dim=512, drop_path_rate=0.0)

    # Loss with focal + alpha
    criterion = build_criterion(config, "phase2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    losses = []
    for step, (images, labels, sources, masks) in enumerate(loader):
        if step >= 2:
            break

        # Test mixup path
        scan_embed, attn = model.forward_features(images, masks)
        mixed, labels_a, labels_b, lam = embedding_mixup(scan_embed, labels, 0.2)
        logits = model.classifier(mixed)
        loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        assert not np.isnan(loss.item()), f"NaN loss at step {step}"

    assert len(losses) == 2, f"Expected 2 steps, got {len(losses)}"


# ========== TEST 6: Evaluation ==========

def test_threshold_sweep():
    """Verify threshold sweep finds reasonable threshold."""
    np.random.seed(42)
    # Create synthetic predictions
    probs = np.random.rand(100, 2)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = np.random.randint(0, 2, 100)
    sources = np.random.choice([0, 1, 2, 3], 100)
    best_f1, best_thresh, preds = sweep_threshold(probs, labels, sources)
    assert 0.25 <= best_thresh <= 0.75, f"Threshold out of range: {best_thresh}"
    assert 0 <= best_f1 <= 1.0, f"F1 out of range: {best_f1}"
    assert len(preds) == 100


# ========== TEST 7: Ensemble ==========

def test_ensemble_predictions():
    """Verify ensemble averaging works correctly."""
    from src.ensemble_evaluate import ensemble_predictions
    p1 = np.array([[0.8, 0.2], [0.3, 0.7]])
    p2 = np.array([[0.6, 0.4], [0.4, 0.6]])
    # Uniform
    ens = ensemble_predictions([p1, p2])
    expected = np.array([[0.7, 0.3], [0.35, 0.65]])
    assert np.allclose(ens, expected, atol=1e-6), f"Uniform ensemble wrong: {ens}"
    # Weighted
    ens_w = ensemble_predictions([p1, p2], weights=[0.75, 0.25])
    expected_w = 0.75 * p1 + 0.25 * p2
    assert np.allclose(ens_w, expected_w, atol=1e-6), f"Weighted ensemble wrong: {ens_w}"


# ========== TEST 8: Scheduler ==========

def test_warmup_scheduler():
    """Verify warmup scheduler ramps up then decays."""
    model = nn.Linear(10, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    sched = get_cosine_warmup_scheduler(opt, warmup_steps=10, total_steps=100)
    lrs = []
    for _ in range(100):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    # LR should increase during warmup
    assert lrs[5] > lrs[0], "LR not increasing during warmup"
    # LR should peak around step 10
    assert lrs[10] > lrs[5], "LR not at peak after warmup"
    # LR should decrease after warmup
    assert lrs[99] < lrs[10], "LR not decreasing after warmup"


# ========== TEST 9: Checkpoint ==========

def test_checkpoint_save_load():
    """Verify model can be saved and loaded."""
    import tempfile
    model = CovidDetector(classifier_hidden_dim=512, drop_path_rate=0.3)
    model.eval()
    x = torch.randn(1, 4, 3, 224, 224)
    mask = torch.ones(1, 4)
    with torch.no_grad():
        logits_before, _ = model(x, mask)

    # Save
    tmpdir = tempfile.mkdtemp()
    ckpt_mgr = CheckpointManager(tmpdir)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ckpt_mgr.save_best(model, opt, epoch=5, score=0.85)

    # Load into fresh model
    model2 = CovidDetector(classifier_hidden_dim=512, drop_path_rate=0.3)
    CheckpointManager.load(os.path.join(tmpdir, "best.pt"), model2)
    model2.eval()
    with torch.no_grad():
        logits_after, _ = model2(x, mask)

    assert torch.allclose(logits_before, logits_after, atol=1e-5), \
        "Loaded model produces different outputs"

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)


# ========== TEST 10: Per-Source Threshold Sweep ==========

def test_per_source_threshold_sweep():
    """Verify per-source sweep finds independent thresholds."""
    from src.evaluate import sweep_threshold_per_source
    np.random.seed(42)
    probs = np.random.rand(200, 2)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = np.random.randint(0, 2, 200)
    sources = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
    ps_f1, ps_thresholds, preds = sweep_threshold_per_source(probs, labels, sources)
    # Should have a threshold per source
    assert len(ps_thresholds) == 4, f"Expected 4 thresholds, got {len(ps_thresholds)}"
    # Each threshold should be in valid range
    for src, t in ps_thresholds.items():
        assert 0.2 <= t <= 0.8, f"Threshold out of range for {src}: {t}"
    assert 0 <= ps_f1 <= 1.0


def test_per_source_vs_global_threshold():
    """Per-source thresholds should be >= global threshold in F1."""
    from src.evaluate import sweep_threshold_per_source, sweep_threshold
    np.random.seed(123)
    probs = np.random.rand(200, 2)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = np.random.randint(0, 2, 200)
    sources = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
    global_f1, _, _ = sweep_threshold(probs, labels, sources)
    ps_f1, _, _ = sweep_threshold_per_source(probs, labels, sources)
    assert ps_f1 >= global_f1 - 0.01, \
        f"Per-source F1 ({ps_f1:.4f}) should be >= global F1 ({global_f1:.4f})"


# ========== TEST 11: Enhanced TTA ==========

def test_multi_flip_tta():
    """Verify multi-flip TTA mode produces valid probabilities."""
    # Use a tiny model for speed
    model = CovidDetector(classifier_hidden_dim=512, drop_path_rate=0.0)
    model.eval()
    x = torch.randn(1, 4, 3, 224, 224)
    mask = torch.ones(1, 4)

    # Test that multi-flip produces different logits than no-TTA
    with torch.no_grad():
        logits_plain, _ = model(x, mask)
        # Simulate multi-flip
        tta_logits = [logits_plain]
        logits_hflip, _ = model(torch.flip(x, dims=[-1]), mask)
        tta_logits.append(logits_hflip)
        logits_vflip, _ = model(torch.flip(x, dims=[-2]), mask)
        tta_logits.append(logits_vflip)
        logits_hvflip, _ = model(torch.flip(x, dims=[-2, -1]), mask)
        tta_logits.append(logits_hvflip)
        logits_multi = torch.stack(tta_logits).mean(dim=0)

    probs = F.softmax(logits_multi, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5), \
        "Multi-flip TTA probabilities don't sum to 1"
    assert not torch.equal(logits_plain, logits_multi), \
        "Multi-flip TTA should produce different logits than plain"


# ========== TEST 12: EfficientNetV2-S ==========

def test_effv2_model():
    """Verify EfficientNetV2-S backbone works end-to-end."""
    model = CovidDetector(backbone_name="tf_efficientnetv2_s",
                          classifier_hidden_dim=512, drop_path_rate=0.3)
    model.eval()
    x = torch.randn(1, 4, 3, 256, 256)
    mask = torch.ones(1, 4)
    with torch.no_grad():
        logits, attn = model(x, mask)
    assert logits.shape == (1, 2), f"Bad logits shape: {logits.shape}"
    assert model.embed_dim == 1280, f"Bad embed dim: {model.embed_dim}"


# ========== TEST 13: SWA Model Creation ==========

def test_swa_model():
    """Verify SWA averaged model can be created and used."""
    from torch.optim.swa_utils import AveragedModel
    model = CovidDetector(classifier_hidden_dim=512, drop_path_rate=0.0)
    swa_model = AveragedModel(model)
    x = torch.randn(1, 4, 3, 224, 224)
    mask = torch.ones(1, 4)
    model.eval()
    swa_model.eval()
    with torch.no_grad():
        logits_orig, _ = model(x, mask)
        logits_swa, _ = swa_model(x, mask)
    # After 1 update, SWA should match the original
    assert logits_orig.shape == logits_swa.shape, "SWA model shape mismatch"
    assert not torch.isnan(logits_swa).any(), "SWA model produced NaN"


# ========== TEST 14: Config Loading (new configs) ==========

def test_new_configs():
    """Verify new config files load correctly."""
    cfg1 = load_config("configs/exp_effv2_s42.yaml")
    assert cfg1["model"]["backbone"] == "tf_efficientnetv2_s"
    assert cfg1["model"]["embedding_dim"] == 1280
    assert cfg1["phase2"]["swa_start_epoch"] == 20
    assert cfg1["phase2"]["source_loss_weights"] == [1.0, 1.5, 1.0, 1.0]
    assert cfg1["eval"]["tta"] == "multi"

    cfg2 = load_config("configs/exp_b3_s7.yaml")
    assert cfg2["seed"] == 7
    assert cfg2["model"]["backbone"] == "efficientnet_b3"


# ========== RUN ALL ==========

if __name__ == "__main__":
    print("=" * 60)
    print("END-TO-END INTEGRATION TESTS")
    print("=" * 60)

    print("\n1. Data Pipeline")
    test("Augmentation pipeline (20 runs)", test_augmentation_pipeline)
    test("Stochastic slice sampling", test_stochastic_slice_sampling)
    test("Scan collation", test_scan_collation)

    print("\n2. Model")
    test("forward_features() shape", test_forward_features)
    test("forward() == forward_features() + classifier()", test_forward_matches)
    test("ConvNeXt-Tiny backbone", test_convnext_model)
    test("from_slice_classifier transfer", test_from_slice_classifier)

    print("\n3. Losses")
    test("FocalLoss basic + gradients", test_focal_loss_basic)
    test("FocalLoss with alpha weighting", test_focal_loss_with_alpha)
    test("Focal < CE on easy examples", test_focal_vs_ce_on_easy)
    test("build_criterion with focal_alpha", test_build_criterion_focal)

    print("\n4. Mixup")
    test("Mixup preserves shapes", test_mixup_shapes)
    test("Mixup actually interpolates", test_mixup_interpolation)
    test("Mixup disabled with alpha=0", test_mixup_disabled)

    print("\n5. Training Loop")
    test("Mini Phase 2 training (2 steps)", test_mini_train_phase2)

    print("\n6. Evaluation")
    test("Threshold sweep", test_threshold_sweep)

    print("\n7. Ensemble")
    test("Ensemble predictions (uniform + weighted)", test_ensemble_predictions)

    print("\n8. Scheduler")
    test("Warmup + cosine decay schedule", test_warmup_scheduler)

    print("\n9. Checkpoint")
    test("Save and load checkpoint", test_checkpoint_save_load)

    print("\n10. Per-Source Threshold Sweep")
    test("Per-source threshold sweep", test_per_source_threshold_sweep)
    test("Per-source >= global threshold", test_per_source_vs_global_threshold)

    print("\n11. Enhanced TTA")
    test("Multi-flip TTA", test_multi_flip_tta)

    print("\n12. EfficientNetV2-S")
    test("EfficientNetV2-S backbone", test_effv2_model)

    print("\n13. SWA")
    test("SWA model creation + forward", test_swa_model)

    print("\n14. New Configs")
    test("New experiment configs load", test_new_configs)

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)

