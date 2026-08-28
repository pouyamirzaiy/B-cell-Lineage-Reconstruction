#!/usr/bin/env python3
"""
Feature-Based ClonalTree with Comb Resolution (UPDATED)
======================================================

Main fix vs your last run:
- primMST in ClonalTree likely casts distances to int (because original uses Hamming distances).
- If distances are < 1, int(distance) becomes 0 -> massive ties -> node contraction/merging.
- We therefore SCALE distances to large integers BEFORE calling primMST.

Outputs:
- ../results/feature_based_outputs/<dataset_name>_final.nk
- ../results/feature_based_outputs/<dataset_name>_final_ascii.txt
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append("../ClonalTree_src")

from BasicSeq import readFastaAbundance
from MSTree import primMST
from BasicTree import getDistances, trimming, editTree


# ---------- IMPORTANT ----------
# Scale distances so even if primMST casts to int, nothing becomes 0.
DIST_SCALE = 1_000_000  # try 1e6 first; if still collapsing, try 1e9
# ------------------------------


def _stable_u01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    x = int(h[:8], 16)  # 32-bit
    return x / 2**32


def _stable_pair_jitter(a: str, b: str, scale: float) -> float:
    s = (a + "||" + b).encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    x = int(h[:8], 16)  # 32-bit
    return (x / 2**32) * scale


def _debug_matrix(D: np.ndarray, name: str):
    n = D.shape[0]
    off = D[np.triu_indices(n, 1)]
    print(f"\n[DEBUG] {name}")
    print(f"  n={n}")
    print(f"  offdiag min/median/max = {float(off.min())} / {float(np.median(off))} / {float(off.max())}")
    print(f"  zero offdiag edges = {int(np.sum(off == 0.0))} out of {off.size}")
    print(f"  unique distances (rounded 1e-12) = {np.unique(np.round(off, 12)).size}")


def _scale_for_primMST(D: np.ndarray, labels, scale=DIST_SCALE):
    """
    Scale distances so primMST (even if it int-casts) keeps information.

    Steps:
    1) multiply by scale
    2) add tiny deterministic pair-jitter (<<1) to break ties
    3) ensure diagonal is 0
    """
    labels = [str(x) for x in labels]
    n = D.shape[0]
    D2 = D * float(scale)

    # tiny pair jitter (sub-integer), so rounding/int-cast gives deterministic tie break
    jitter = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            # jitter in [0, 0.999...) so it survives rounding but doesn't dominate
            eps = _stable_pair_jitter(labels[i], labels[j], 0.999)
            jitter[i, j] = eps
            jitter[j, i] = eps

    D2 = D2 + jitter
    np.fill_diagonal(D2, 0.0)

    # Debug: what happens if someone casts to int?
    off = D2[np.triu_indices(n, 1)]
    zeros_if_int = int(np.sum(np.floor(off).astype(np.int64) == 0))
    print(f"[DEBUG] after scaling x{scale}: zeros_if_int_cast = {zeros_if_int} out of {off.size}")

    return D2


class FeatureBasedClonalTree:
    def __init__(self):
        self.data_path = Path("../data/sarah_kaveh_materials/feature 2")
        self.results_path = Path("../results")
        (self.results_path / "feature_based_outputs").mkdir(parents=True, exist_ok=True)

        print("🌳 FEATURE-BASED CLONALTREE")
        print("=" * 40)
        print(f"📁 Data path: {self.data_path}")
        print()

        self.features_data_full = {}
        self.features_data_base = {}
        self.feature_names = []

    def load_features_for_dataset(self, dataset_name: str):
        dataset_num = dataset_name.split("_")[1]
        feature_file = self.data_path / f"feature_{dataset_num}.csv"
        if not feature_file.exists():
            print(f"❌ Feature file not found: {feature_file}")
            return None

        df = pd.read_csv(feature_file)
        id_col = df.columns[0]
        feature_names = df.columns[1:].tolist()

        full_map = {}
        base_map = {}
        for _, row in df.iterrows():
            seq_id = str(row[id_col]).strip()
            feats = row.iloc[1:].to_dict()
            full_map[seq_id] = feats
            base_id = seq_id.split("@")[0]
            if base_id not in base_map:
                base_map[base_id] = feats

        self.features_data_full = full_map
        self.features_data_base = base_map
        self.feature_names = feature_names

        print(f"✅ Loaded {len(full_map)} feature rows with {len(feature_names)} features")
        print(f"   (base-id map size: {len(base_map)})")
        return full_map, base_map, feature_names

    def _get_feature_value(self, label: str, feature_name: str):
        lab = str(label).strip()
        base = lab.split("@")[0]
        if lab == "naive@1":
            lab = "naive"
            base = "naive"
        if lab in self.features_data_full:
            return self.features_data_full[lab].get(feature_name, np.nan)
        if base in self.features_data_base:
            return self.features_data_base[base].get(feature_name, np.nan)
        return np.nan

    def create_single_feature_adjacency(self, labels, feature_name, normalize="rank"):
        n = len(labels)
        vals = np.empty(n, dtype=float)

        for i, lab in enumerate(labels):
            v = self._get_feature_value(lab, feature_name)
            try:
                vals[i] = float(v)
            except Exception:
                vals[i] = np.nan

        if np.isnan(vals).any():
            m = np.nanmedian(vals)
            if np.isnan(m):
                m = 0.0
            vals = np.nan_to_num(vals, nan=m)

        if normalize == "z":
            mu = float(np.mean(vals))
            sd = float(np.std(vals)) or 1.0
            vals = (vals - mu) / sd
        elif normalize == "rank":
            order = np.argsort(vals, kind="mergesort")
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(n, dtype=float) / max(1, n - 1)
            vals = ranks

        D = np.abs(vals.reshape(-1, 1) - vals.reshape(1, -1))
        np.fill_diagonal(D, 0.0)
        return D

    def create_multi_feature_adjacency(
        self,
        labels,
        feature_list,
        normalize="z",
        label_jitter_scale=1e-6,   # keeps vectors unique; very small
    ):
        labels = [str(x) for x in labels]
        n = len(labels)
        k = len(feature_list)
        X = np.empty((n, k), dtype=float)
        missing_raw = 0

        for i, lab in enumerate(labels):
            for j, f in enumerate(feature_list):
                v = self._get_feature_value(lab, f)
                try:
                    X[i, j] = float(v)
                except Exception:
                    X[i, j] = np.nan
                    missing_raw += 1

        # impute by median per column
        for j in range(k):
            col = X[:, j]
            mask = np.isfinite(col)
            if not np.any(mask):
                X[:, j] = 0.0
                continue
            med = float(np.median(col[mask]))
            col[~mask] = med
            X[:, j] = col

        if normalize == "z":
            mu = X.mean(axis=0)
            sd = X.std(axis=0)
            sd[sd == 0] = 1.0
            X = (X - mu) / sd

        # per-label tiny jitter in feature space (does NOT dominate)
        for i, lab in enumerate(labels):
            for j, f in enumerate(feature_list):
                u = _stable_u01(f"{lab}::{f}")
                X[i, j] += (u - 0.5) * label_jitter_scale

        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(D, 0.0)

        print(f"[DEBUG] multi-feature adjacency raw: n={n}, k={k}, missing_raw_values={missing_raw}")
        _debug_matrix(D, "raw multi-feature D")

        return D

    def rank_features_global(self, labels_in_use):
        rows = []
        for lab in labels_in_use:
            base = str(lab).split("@")[0]
            feats = None
            if base in self.features_data_base:
                feats = self.features_data_base[base]
            elif str(lab) in self.features_data_full:
                feats = self.features_data_full[str(lab)]
            if feats is not None:
                rows.append([feats.get(f, np.nan) for f in self.feature_names])

        if not rows:
            return self.feature_names[:]

        X = np.array(rows, dtype=float)
        scores = []
        for j, f in enumerate(self.feature_names):
            col = X[:, j]
            col = col[np.isfinite(col)]
            if len(col) < 3:
                scores.append((-np.inf, f))
                continue
            mu = float(np.mean(col))
            sd = float(np.std(col))
            cv = sd / (abs(mu) + 1e-12)
            uniq = len(np.unique(col))
            ties = len(col) - uniq
            score = (cv * uniq) / (ties + 1.0)
            scores.append((score, f))

        scores.sort(reverse=True, key=lambda x: x[0])
        return [f for _, f in scores]

    def run_feature_based_clonaltree(
        self,
        dataset_name,
        fasta_file,
        useAbundance=True,
        revision=False,
        trim=False,
        initial_k_features=10,
    ):
        if self.load_features_for_dataset(dataset_name) is None:
            return None

        labels, root, arraySeqs, abundance, dico = readFastaAbundance(fasta_file)
        if not isinstance(abundance, dict):
            raise TypeError("Expected abundance dict.")

        total_reads = int(sum(abundance.values()))
        print("\n" + "=" * 80)
        print("DEBUG: readFastaAbundance() outputs")
        print("=" * 80)
        print(f"[labels] len(unique) = {len(labels)}")
        print(f"[abundance] total_reads = {total_reads}")
        print(f"[root] {repr(root)}")

        # Keep ALL labels (missing features will be imputed)
        missing_in_features = []
        present = 0
        for lab in labels:
            base = str(lab).split("@")[0]
            if (str(lab) in self.features_data_full) or (base in self.features_data_base) or (lab == "naive@1" and "naive" in self.features_data_base):
                present += 1
            else:
                missing_in_features.append(lab)

        print("\n" + "-" * 80)
        print(f"[features] present for {present}/{len(labels)} labels")
        print(f"[features] missing for {len(missing_in_features)} labels (kept, will be imputed)")
        if missing_in_features:
            print("  example missing:", missing_in_features[:10])

        used_labels = list(labels)
        used_abundance = {lab: abundance.get(lab, abundance.get(str(lab).split("@")[0], 1)) for lab in used_labels}

        ranked = self.rank_features_global([str(l).split("@")[0] for l in used_labels])
        if not ranked:
            print("❌ No ranked features.")
            return None

        k0 = max(2, int(initial_k_features))
        base_features = ranked[:k0]
        print("\nUsing MULTI-feature initial MST:")
        print(f"  k={k0}")
        print("  base_features:", base_features)

        D_raw = self.create_multi_feature_adjacency(
            used_labels,
            base_features,
            normalize="z",
            label_jitter_scale=1e-6,
        )

        # >>> KEY: scale for primMST <<<
        adjMatrix = _scale_for_primMST(D_raw, used_labels, scale=DIST_SCALE)
        _debug_matrix(adjMatrix, f"scaled adjacency passed to primMST (x{DIST_SCALE})")

        # root handling
        if isinstance(root, int) and 0 <= root < len(used_labels):
            root_idx = root
        else:
            root_idx = 0
            if root in used_labels:
                root_idx = used_labels.index(root)

        tree, infoTree = primMST(adjMatrix, root_idx, used_labels, used_abundance, useAbundance)

        if trim:
            tree = trimming(tree, used_labels, adjMatrix)
        if revision:
            tree = editTree(tree, adjMatrix, used_labels)

        infoTree = getDistances(tree)

        leaf_names = tree.get_leaf_names()
        print("\n" + "-" * 80)
        print(f"[tree] leaves = {len(leaf_names)} ; expected = {len(used_labels)}")
        missing_from_tree = sorted(set(map(str, used_labels)) - set(map(str, leaf_names)))
        if missing_from_tree:
            print("[tree][WARN] leaf mismatch detected!")
            print("  missing_from_tree (first 30):", missing_from_tree[:30])

        # Only rename leaves if leaf base exists in abundance (avoid turning weird merged names into @1)
        for leaf in tree.iter_leaves():
            base = leaf.name.split("@")[0]
            if base in used_abundance:
                leaf.name = f"{base}@{used_abundance[base]}"

        out_dir = self.results_path / "feature_based_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_newick = out_dir / f"{dataset_name}_final.nk"
        tree.write(outfile=str(out_newick), format=1)
        print("Saved final Newick:", out_newick)

        out_ascii = out_dir / f"{dataset_name}_final_ascii.txt"
        out_ascii.write_text(tree.get_ascii(show_internal=True) + "\n", encoding="utf-8")
        print("Saved final ASCII:", out_ascii)

        return tree, infoTree


def main():
    clonaltree = FeatureBasedClonalTree()

    # Base folder that contains all LLC fasta files
    fasta_dir = Path(r"E:\clonaltree\modified\LLC")

    # Common filename patterns to try (in case your files differ slightly)
    patterns = [
        "LLC_dataset{num:02d}_1_200_sequences.aln.fa",
        "LLC_dataset{num:02d}_1_200_sequences.fa",
        "LLC_dataset{num:02d}.aln.fa",
        "LLC_dataset{num:02d}.fa",
        "dataset_{num:02d}.aln.fa",
        "dataset_{num:02d}.fa",
    ]

    total = 57
    ok = 0
    skipped = 0
    failed = 0

    for num in range(1, total + 1):
        dataset_name = f"dataset_{num:02d}"

        # find the first existing fasta that matches any known pattern
        fasta_file = None
        for pat in patterns:
            candidate = fasta_dir / pat.format(num=num)
            if candidate.exists():
                fasta_file = candidate
                break

        if fasta_file is None:
            print(f"\n⏭️  SKIP {dataset_name}: FASTA not found (tried {len(patterns)} patterns in {fasta_dir})")
            skipped += 1
            continue

        print("\n" + "=" * 90)
        print(f"▶ Running {dataset_name}")
        print(f"   FASTA: {fasta_file}")
        print("=" * 90)

        try:
            res = clonaltree.run_feature_based_clonaltree(
                dataset_name=dataset_name,
                fasta_file=str(fasta_file),
                useAbundance=True,
                revision=False,
                trim=False,
                initial_k_features=10,
            )
            if res:
                ok += 1
            else:
                failed += 1
                print(f"❌ {dataset_name}: run_feature_based_clonaltree returned None")
        except Exception as e:
            failed += 1
            print(f"❌ {dataset_name}: exception -> {type(e).__name__}: {e}")

    print("\n" + "#" * 90)
    print("ALL DATASETS FINISHED")
    print(f"✅ success: {ok}")
    print(f"⏭️ skipped (missing fasta): {skipped}")
    print(f"❌ failed (errors/None): {failed}")
    print("#" * 90)


if __name__ == "__main__":
    main()
