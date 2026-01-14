import numpy as np
import pandas as pd
import torch
from torch.utils import data
import torch.nn.functional as F
import json
from torchvision import transforms
from random import random
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

import grating_dataset as ds
import utils.torch_utils as tu
import data.augmentations_003 as aug3
import ml_tracker.utilities.eva_embeddings as ev

from occlusion import KeepOneRandomPatchOnly, KeepTwoRandomPatchesSplitMiddle
## Settings that increase speed while lowering accuracy probably <0.1 pp and making not reproducible to the number
# Enable TF32 for matmul (GEMM)
torch.set_float32_matmul_precision('high')  # options: 'highest' (disable TF32), 'high' (enable TF32), 'medium'

# Also enable TF32 for cuDNN convolutions (useful for CNNs / 3D CNNs)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE="cuda"


def get_drifting_grating(sel):
    # 001
    if sel == "single":
        drifting_grating_ds = ds.DatasetDriftingGrating(
            parameter_list="DEFAULT",
            y_par=["spatial_freq", "orientation_deg", "temporal_freq", "contrast", "color"],
            color_dim=3,
            color_as_rgb=False)
        drifting_grating_dl = data.DataLoader(drifting_grating_ds,
                                              batch_size=128,
                                              shuffle=False,
                                              num_workers=10,
                                              pin_memory=True)
        return drifting_grating_ds, drifting_grating_dl
    elif sel == "double":
        double_drifting_grating_ds = ds.DatasetDriftingGrating(
            parameter_list="DEFAULT_TWO_REGION",
            y_par=["spatial_freq_left",
                "spatial_freq_right",
                "sf_delta",
                "sf_higher_side",
                "spatial_freq_match",
                "orientation_deg_left",
                "orientation_deg_right",
                "orientation_delta",
                "orientation_match"],
            color_dim=3,
            color_as_rgb=False)
        double_drifting_grating_dl = data.DataLoader(double_drifting_grating_ds,
                                              batch_size=128,
                                              shuffle=False,
                                              num_workers=10,
                                              pin_memory=True)
        return double_drifting_grating_ds, double_drifting_grating_dl
    else:
        raise ValueError("Selection for Drifting Grating not recognized")

def get_drifting_grating_masked(sel):
    #002
    augs = transforms.Compose([
                aug3.TubeMaskingGeneratorTorch(window_size=(8,16,16), mask_ratio=0.9)
            ])
    if sel == "single":
        drifting_grating_ds = ds.DatasetDriftingGrating(
            parameter_list="DEFAULT",
            y_par=["spatial_freq", "orientation_deg", "temporal_freq", "contrast", "color"],
            color_dim=3,
            color_as_rgb=False,
            transform=augs)
        drifting_grating_dl = data.DataLoader(drifting_grating_ds,
                                              batch_size=128,
                                              shuffle=False,
                                              num_workers=10,
                                              pin_memory=True)
        return drifting_grating_ds, drifting_grating_dl
    elif sel == "double":
        double_drifting_grating_ds = ds.DatasetDriftingGrating(
            parameter_list="DEFAULT_TWO_REGION",
            y_par=["spatial_freq_left",
                "spatial_freq_right",
                "sf_delta",
                "sf_higher_side",
                "spatial_freq_match",
                "orientation_deg_left",
                "orientation_deg_right",
                "orientation_delta",
                "orientation_match"],
            color_dim=3,
            color_as_rgb=False,
            transform=augs)
        double_drifting_grating_dl = data.DataLoader(double_drifting_grating_ds,
                                              batch_size=128,
                                              shuffle=False,
                                              num_workers=10,
                                              pin_memory=True)
        return double_drifting_grating_ds, double_drifting_grating_dl
    else:
        raise ValueError("Selection for Drifting Grating not recognized")

def get_drifting_grating_occl(sel):
    # 005
    if sel == "single":
        augs = transforms.Compose([
                    KeepOneRandomPatchOnly()
                ])
        drifting_grating_ds = ds.DatasetDriftingGrating(
            parameter_list="DEFAULT",
            y_par=["spatial_freq", "orientation_deg", "temporal_freq", "contrast", "color"],
            color_dim=3,
            color_as_rgb=False,
            transform=augs)
        drifting_grating_dl = data.DataLoader(drifting_grating_ds,
                                              batch_size=128,
                                              shuffle=False,
                                              num_workers=10,
                                              pin_memory=True)
        return drifting_grating_ds, drifting_grating_dl
    elif sel == "double":
        augs2 = transforms.Compose([
            KeepTwoRandomPatchesSplitMiddle()
                ])

        double_drifting_grating_ds = ds.DatasetDriftingGrating(
            parameter_list="DEFAULT_TWO_REGION",
            y_par=["spatial_freq_left",
                "spatial_freq_right",
                "sf_delta",
                "sf_higher_side",
                "spatial_freq_match",
                "orientation_deg_left",
                "orientation_deg_right",
                "orientation_delta",
                "orientation_match"],
            color_dim=3,
            color_as_rgb=False,
            transform=augs2)
        double_drifting_grating_dl = data.DataLoader(double_drifting_grating_ds,
                                              batch_size=128,
                                              shuffle=False,
                                              num_workers=10,
                                              pin_memory=True)
        return double_drifting_grating_ds, double_drifting_grating_dl
    else:
        raise ValueError("Selection for Drifting Grating not recognized")


def return_agg_encodings(mdl, dl, blocks=4, transform=None):
    # 001, 003, 005, 007, 008
    emb_collection = {k: [] for k in range(blocks)}
    z = {}
    ys = []

    with torch.no_grad():
        mdl.eval()
        for x, y in dl:
            if isinstance(x, tuple) or isinstance(x, list):
                x = x[0]
            if transform is not None:
                x = transform(x)
            torch.compiler.cudagraph_mark_step_begin()
            batch, y = tu.move_to_device(x, DEVICE), tu.move_to_device(y, DEVICE)
            embeddings = mdl.encode(batch)
            for nr, embedding in enumerate(embeddings):
                emb_collection[nr].append(embedding.detach().cpu().mean(axis=1))
            ys.append(y.detach().cpu())
        for k,v in emb_collection.items():
            z[k] = torch.cat(v).numpy()
        y_cat = torch.cat(ys).numpy()
        return z, y_cat

def return_encodings_sample(mdl, dl, blocks=4):
    # 004
    emb_collection = {k: [] for k in range(blocks)}
    z = {}
    ys = []

    with torch.no_grad():
        mdl.eval()

        ratio = 8/len(dl)

        for x, y in dl:
            if random() > ratio:
                continue
            if isinstance(x, tuple) or isinstance(x, list):
                x = x[0]
            torch.compiler.cudagraph_mark_step_begin()
            batch, y = tu.move_to_device(x, DEVICE), tu.move_to_device(y, DEVICE)
            embeddings = mdl.encode(batch)
            for nr, embedding in enumerate(embeddings):
                emb_collection[nr].append(embedding.detach().cpu())
            ys.append(y.detach().cpu())
        for k,v in emb_collection.items():
            z[k] = torch.cat(v)
        y_cat = torch.cat(ys)
        return z, y_cat

def cos_patch_similarity(E: torch.Tensor, *, reduce: str = "none", eps: float = 1e-12):
    # 004
    """
    E: (B, P, D) patch/token embeddings
    Returns:
      - if reduce="none": (B,) diversity per image
      - if reduce="mean": scalar mean over batch
      - if reduce="sum":  scalar sum over batch
    Similarity = mean_{i!=j} cos(e_i, e_j)
    """

    # print(E.mean(), E.min(), E.max(), E.std())

    if E.ndim != 3:
        raise ValueError(f"Expected (B,P,D), got {tuple(E.shape)}")
    B, P, D = E.shape
    if P < 2:
        out = E.new_zeros((B,))
        return out.mean() if reduce == "mean" else out.sum() if reduce == "sum" else out

    # per-token standardization (LayerNorm without affine)
    E0 = F.layer_norm(E, (D,), weight=None, bias=None, eps=eps)

    # print(E0.mean(), E0.min(), E0.max(), E0.std())

    En = F.normalize(E0, dim=-1, eps=eps)          # (B, P, D)

    # print(En.mean(), En.min(), En.max(), En.std())

    S = En @ En.transpose(-1, -2)                # (B, P, P)

    diag_sum = S.diagonal(dim1=-2, dim2=-1).sum(dim=-1)     # (B,)
    off_diag_mean = (S.sum(dim=(-1, -2)) - diag_sum) / (P * (P - 1))  # (B,)

    similarity = off_diag_mean              # (B,)

    if reduce == "none":
        return similarity
    if reduce == "mean":
        return similarity.mean()
    if reduce == "sum":
        return similarity.sum()
    raise ValueError("reduce must be one of: 'none', 'mean', 'sum'")

def log_regr_multi_target_cv(z, y, n_splits=3):
    # 001, 005
    results = {}
    y_cat = y.astype(int)

    n_targets = y.shape[1]

    # Base model (no multi_class -> no FutureWarning)
    log_reg = LogisticRegression(
        max_iter=1000,
    )

    # Multi-output vs single-output
    if n_targets > 1:
        clf_cat = make_pipeline(
            StandardScaler(),
            MultiOutputClassifier(log_reg)
        )
    else:
        clf_cat = make_pipeline(
            StandardScaler(),
            log_reg
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=0
    )

    overall_scores = []
    per_target_scores = np.zeros((n_splits, n_targets))

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(z, y_cat[:, 0])):
        X_train, X_test = z[train_idx], z[test_idx]
        y_train, y_test = y_cat[train_idx], y_cat[test_idx]

        # ----- Handle y shapes differently for single vs multi target -----
        if n_targets == 1:
            # Flatten to 1D for LogisticRegression
            y_train_fit = y_train.ravel()      # (n_train,)
            y_test_score = y_test.ravel()      # (n_test,)
        else:
            # Keep 2D for MultiOutputClassifier
            y_train_fit = y_train              # (n_train, n_targets)
            y_test_score = y_test              # (n_test, n_targets)

        # Train on this fold
        clf_cat.fit(X_train, y_train_fit)

        # Overall accuracy (all outputs correct for n_targets > 1)
        overall_acc = clf_cat.score(X_test, y_test_score)
        overall_scores.append(overall_acc)

        # Per-target accuracy
        y_pred = clf_cat.predict(X_test)

        for j in range(n_targets):
            if n_targets == 1:
                # y_pred is 1D, y_test_score is 1D
                y_true_sel = y_test_score
                y_pred_sel = y_pred
            else:
                # y_pred and y_test are 2D: (n_samples, n_targets)
                y_true_sel = y_test[:, j]
                y_pred_sel = y_pred[:, j]

            acc_j = accuracy_score(y_true_sel, y_pred_sel)
            per_target_scores[fold_idx, j] = acc_j

    overall_scores = np.array(overall_scores)

    # Store aggregated results
    results["cv_overall_accuracy_mean"] = overall_scores.mean()
    results["cv_overall_accuracy_std"] = overall_scores.std()

    for j in range(n_targets):
        mean_j = per_target_scores[:, j].mean()
        std_j = per_target_scores[:, j].std()
        results[f"cv_target_{j}_accuracy_mean"] = mean_j
        results[f"cv_target_{j}_accuracy_std"] = std_j

    return results

def log_regr_multi_target(z_train, y_train, z_test, y_test):
    # 003, 006
    """
    Train a (multi-output) logistic regression probe on training data and
    evaluate on a separate test set.

    Parameters
    ----------
    z_train : np.ndarray, shape (n_train_samples, n_features)
        Training feature matrix.
    y_train : np.ndarray, shape (n_train_samples, n_targets)
        Training labels (integer-coded). For single-target, use shape (n_train, 1).
    z_test : np.ndarray, shape (n_test_samples, n_features)
        Test feature matrix.
    y_test : np.ndarray, shape (n_test_samples, n_targets)
        Test labels (integer-coded), same number of targets as y_train.

    Returns
    -------
    results : dict
        Dictionary with keys:
            - "cv_overall_accuracy_mean"
            - "cv_overall_accuracy_std"   (0.0, since only one split)
            - "cv_target_{j}_accuracy_mean"
            - "cv_target_{j}_accuracy_std" (0.0 per target)
        These names are kept for compatibility with the original CV version.
    """
    results = {}

    # Ensure arrays and cast labels to int
    z_train = np.asarray(z_train)
    z_test = np.asarray(z_test)
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)

    # Make sure labels are 2D: (n_samples, n_targets)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    n_targets = y_train.shape[1]
    assert y_test.shape[1] == n_targets, "y_train and y_test must have same #targets"

    # Base model (no multi_class -> no FutureWarning)
    log_reg = LogisticRegression(max_iter=1000)

    # Multi-output vs single-output
    if n_targets > 1:
        clf_cat = make_pipeline(
            StandardScaler(),
            MultiOutputClassifier(log_reg)
        )
    else:
        clf_cat = make_pipeline(
            StandardScaler(),
            log_reg
        )

    # ----- Handle y shapes differently for single vs multi target -----
    if n_targets == 1:
        # Flatten to 1D for LogisticRegression
        y_train_fit = y_train.ravel()   # (n_train,)
        y_test_score = y_test.ravel()   # (n_test,)
    else:
        # Keep 2D for MultiOutputClassifier
        y_train_fit = y_train           # (n_train, n_targets)
        y_test_score = y_test           # (n_test, n_targets)

    # ----- Train on training set -----
    clf_cat.fit(z_train, y_train_fit)

    # ----- Overall accuracy (all outputs correct for n_targets > 1) -----
    overall_acc = clf_cat.score(z_test, y_test_score)

    # ----- Per-target accuracy -----
    y_pred = clf_cat.predict(z_test)

    per_target_scores = np.zeros((1, n_targets), dtype=float)

    for j in range(n_targets):
        if n_targets == 1:
            # y_pred is 1D, y_test_score is 1D
            y_true_sel = y_test_score
            y_pred_sel = y_pred
        else:
            # y_pred and y_test are 2D: (n_samples, n_targets)
            y_true_sel = y_test[:, j]
            y_pred_sel = y_pred[:, j]

        acc_j = accuracy_score(y_true_sel, y_pred_sel)
        per_target_scores[0, j] = acc_j

    # ----- Store aggregated results -----
    overall_scores = np.array([overall_acc])

    # For compatibility with CV version: "mean" over 1 split, std = 0.
    results["cv_overall_accuracy_mean"] = float(overall_scores.mean())

    for j in range(n_targets):
        mean_j = float(per_target_scores[:, j].mean())  # single split -> just acc_j
        results[f"cv_target_{j}_accuracy_mean"] = mean_j

    return results

def run_single_probe(solver: str, k, z_train, y_train, z_test, y_test):
    """
    Run the appropriate probe for a single (solver, k) pair.
    Returns (solver, k, result) so the caller can route the output.
    """
    if solver == "logr":
        res = log_regr_multi_target(z_train[k], y_train, z_test[k], y_test)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    return solver, k, res

def return_loss(mdl, dl, blocks=4):
    # 002
    emb_collection = {k: [] for k in range(blocks)}
    z = {}

    with torch.no_grad():
        mdl.eval()
        for x, y in dl:
            torch.compiler.cudagraph_mark_step_begin()
            batch, y = tu.move_to_device(x, DEVICE), tu.move_to_device(y, DEVICE)
            embeddings = mdl.detached_forward(batch[0], batch[1])
            for nr, embedding in enumerate(embeddings):
                emb_collection[nr].append(embedding.detach().cpu())
        for k,v in emb_collection.items():
            z[k] = float(torch.stack(v).mean().numpy())
        return z

def build_metrics(k, z, y):
    z = z[k]
    results = {}
    results.update(ev._distribution_scale_metrics(z))
    results.update(ev._geometry_spectrum_metrics(z))
    results.update(ev._uniformity_metric(z))

    if y.shape[1] > 1:
        per_target = [ev._retrieval_metrics(z, y[:, j]) for j in range(y.shape[1])]

        # Average same keys across targets (nan-safe)
        avg_retrieval = {}
        keys = per_target[0].keys() if per_target else []
        for key in keys:
            vals = np.array([d.get(key, np.nan) for d in per_target], dtype=float)
            avg_retrieval[key] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")

        results.update(avg_retrieval)
    else:
        results.update(ev._retrieval_metrics(z, y))

    return k, results
