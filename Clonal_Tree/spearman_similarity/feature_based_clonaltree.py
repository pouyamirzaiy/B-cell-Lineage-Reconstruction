#!/usr/bin/env python3
"""
Feature-Based ClonalTree with Comb Resolution
============================================

This script creates a feature-based version of ClonalTree that:
1. Uses the EXACT same ClonalTree algorithm (MST with abundance handling)
2. Only changes the distance calculation from Hamming to feature-based
3. Adds iterative comb resolution using the same algorithm
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime

# Add the ClonalTree source to the path
sys.path.append('/Users/nikaabdollahi/Desktop/Reza_phylo_project/ClonalTree_code/ClonalTree_src')

# Import the original ClonalTree functions
from BasicSeq import readFastaAbundance, createAdjMatrix, hamming_distance

from MSTree import primMST, chooseBestNode, aminIndex, aminIndexFirstFound, addNodeTree, correctMatrix
from BasicTree import getDistances, trimming, editTree
from ete3 import Tree

class FeatureBasedClonalTree:
    """
    Feature-based version of ClonalTree that uses the exact same algorithm
    but with feature-based distances instead of Hamming distances.
    """
    
    def __init__(self):
        self.data_path = Path("../data/sarah_kaveh_materials/feature 2")
        self.results_path = Path("../results")
        
        print("🌳 FEATURE-BASED CLONALTREE")
        print("=" * 40)
        print(f"📁 Data path: {self.data_path}")
        print()
    
    def load_features_for_dataset(self, dataset_name):
        """Load feature data for a specific dataset"""
        dataset_num = dataset_name.split('_')[1]  # Extract number from "dataset_54"
        feature_file = self.data_path / f"feature_{dataset_num}.csv"
        
        if not feature_file.exists():
            print(f"❌ Feature file not found: {feature_file}")
            return None
        
        try:
            # Read the feature file
            df = pd.read_csv(feature_file)
            
            # Extract features (assuming first column is sequence ID, rest are features)
            features_data = {}
            feature_names = df.columns[1:].tolist()  # Skip first column (sequence ID)
            
            for _, row in df.iterrows():
                seq_id = row.iloc[0]  # First column is sequence ID
                features = row.iloc[1:].values  # Rest are features
                features_data[seq_id] = dict(zip(feature_names, features))
            
            print(f"✅ Loaded {len(features_data)} sequences with {len(feature_names)} features")
            return features_data, feature_names
            
        except Exception as e:
            print(f"❌ Error loading features: {e}")
            return None
    
    def create_feature_adjacency_matrix(self, arraySeqs, labels, features_data, feature_names):
        """Create adjacency matrix using feature-based distances instead of Hamming"""
        n = len(arraySeqs)
        adjMatrix = np.zeros((n, n))
        
        print(f"    Creating feature-based adjacency matrix ({n}x{n})...")
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    adjMatrix[i][j] = 0.0
                else:
                    # Get feature vectors for both sequences
                    seq_i = labels[i]
                    seq_j = labels[j]
                    
                    if seq_i in features_data and seq_j in features_data:
                        # Calculate feature-based distance
                        distance = self.calculate_feature_distance(
                            features_data[seq_i], 
                            features_data[seq_j], 
                            feature_names
                        )
                        adjMatrix[i][j] = distance
                    else:
                        # Fallback to Hamming distance if features not available
                        adjMatrix[i][j] = hamming_distance(arraySeqs[i], arraySeqs[j])
        
        return adjMatrix
    def create_single_feature_adjacency(self, labels, features_data, feature_name, normalize="rank"):
        n = len(labels)
        vals = []
        for lab in labels:
            # Handle naive@1 = naive mapping
            lookup_lab = lab
            if lab == "naive@1" and "naive" in features_data:
                lookup_lab = "naive"
            
            v = features_data.get(lookup_lab, {}).get(feature_name, np.nan)
            vals.append(np.nan if v is None else float(v))
        vals = np.array(vals, dtype=float)

        # Handle missing: impute with median (or fall back to 0.0)
        if np.isnan(vals).any():
            m = np.nanmedian(vals) if not np.isnan(np.nanmedian(vals)) else 0.0
            vals = np.nan_to_num(vals, nan=m)

        # Optional normalization to improve scale comparability
        if normalize == "z":
            mu, sd = np.mean(vals), np.std(vals) or 1.0
            vals = (vals - mu) / sd
        elif normalize == "rank":
            # rank-transform to [0,1]
            order = vals.argsort()
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(n) / max(1, n - 1)
            vals = ranks

        # L1 distances on the 1D feature
        adj = np.abs(vals.reshape(-1, 1) - vals.reshape(1, -1))
        np.fill_diagonal(adj, 0.0)
        return adj

    def calculate_feature_distance(self, features1, features2, feature_names):
        """Calculate distance between two feature vectors"""
        distances = []
        
        for feature_name in feature_names:
            if feature_name in features1 and feature_name in features2:
                val1 = features1[feature_name]
                val2 = features2[feature_name]
                
                # Handle different data types
                try:
                    val1 = float(val1)
                    val2 = float(val2)
                    distances.append(abs(val1 - val2))
                except (ValueError, TypeError):
                    # If not numeric, use 0 distance (same value)
                    distances.append(0.0)
            else:
                distances.append(0.0)
        
        # Return the sum of all feature distances
        return sum(distances)
    
    def identify_comb_nodes(self, tree):
        """Identify comb nodes in the tree (nodes with 3+ children)"""
        comb_nodes = []
        
        for node in tree.traverse():
            if len(node.children) >= 3:
                comb_nodes.append(node)
        
        return comb_nodes
    
    def resolve_comb_node(self, comb_node, features_data, feature_names, abundance):
        """Resolve a comb node by re-computing distances using different features"""
        if len(comb_node.children) < 3:
            return comb_node  # No need to resolve
        
        print(f"    Resolving comb node with {len(comb_node.children)} children...")
        
        # Get all sequences in this subtree
        sequences = []
        labels = []
        for child in comb_node.children:
            if child.is_leaf():
                sequences.append(str(child.sequence) if hasattr(child, 'sequence') else '')
                labels.append(child.name)
            else:
                # For internal nodes, we need to get the sequence somehow
                # This is a simplified approach
                sequences.append('')
                labels.append(child.name)
        
        # Create new adjacency matrix for this subtree
        adjMatrix = self.create_feature_adjacency_matrix(sequences, labels, features_data, feature_names)
        
        # Use ClonalTree algorithm to rebuild this subtree
        # This is a simplified version - in practice, you'd need to handle the tree structure more carefully
        subtree, _ = primMST(adjMatrix, 0, labels, abundance, useAb=True)
        
        return subtree

    def rank_features_global(self, features_data, feature_names):
        # Build a matrix (seqs x features)
        seqs = list(features_data.keys())
        X = []
        for s in seqs:
            row = []
            for f in feature_names:
                v = features_data[s].get(f, np.nan)
                row.append(np.nan if v is None else float(v))
            X.append(row)
        X = np.array(X, dtype=float)

        # Score each feature by (z-scored CV + z-scored #unique - z-scored #ties)
        scores = []
        for j, f in enumerate(feature_names):
            col = X[:, j]
            col = col[~np.isnan(col)]
            if len(col) < 3:
                scores.append((-np.inf, f))
                continue
            mu = np.mean(col)
            sd = np.std(col)
            cv = (sd / (abs(mu) + 1e-12))
            uniq = len(np.unique(col))
            ties = len(col) - uniq

            # raw components
            scores.append((cv, uniq, -ties, f))

        if not scores:
            return feature_names[:]  # fallback

        # z-score each component across features
        comps = np.array([[s[0], s[1], s[2]] for s in scores], dtype=float)
        comps = (comps - comps.mean(axis=0)) / (comps.std(axis=0) + 1e-12)
        total = comps.sum(axis=1)

        ranked = [scores[i][3] for i in np.argsort(-total)]  # descending
        return ranked
    def rebuild_subtree_with_feature(self, labels_subset, feature_name, abundance, root_name=None):
        # Keep order stable
        idx_map = {lab: i for i, lab in enumerate(labels_subset)}
        adj = self.create_single_feature_adjacency(labels_subset, self.features_data, feature_name, normalize="rank")

        # Choose a root index: if root_name provided and present, use it; else 0
        root_idx = 0 if (root_name is None or root_name not in idx_map) else idx_map[root_name]

        # Abundance vector sliced to subset order (default to 1 if missing)
        abund = []
        for lab in labels_subset:
            abund.append(abundance.get(lab, 1) if isinstance(abundance, dict) else 1)
        # primMST signature in your code: primMST(adjMatrix, root, labels, abundance, useAb=True)
        subtree, _ = primMST(adj, root_idx, labels_subset, abund, True)
        return subtree

    def graft_subtree(self, global_tree, comb_node, new_subtree):
        # We assume comb_node’s name becomes the root anchor for the new subtree.
        # If new_subtree.root has a different name, we’ll attach its children under comb_node.
        # 1) Collect which leaves are under comb_node now
        old_children = list(comb_node.children)
        for ch in old_children:
            comb_node.remove_child(ch)

        # 2) If new_subtree root corresponds to comb_node, move its children; else attach whole subtree under comb_node
        # We’ll reattach all children of new_subtree root into comb_node, preserving internal structure.
        # (ete3 supports copy: new_subtree.write(...) then read, but we can also clone nodes manually.)
        # Easiest: transfer children recursively by creating new nodes under comb_node.
        def clone_into(target_node, source_node):
            # Skip creating a new wrapper for the source root if we’re attaching children only
            for child in source_node.children:
                new_child = target_node.add_child(name=child.name)
                # copy attributes if you had any (e.g., abundance, sequence)
                for k, v in child.__dict__.items():
                    if k.startswith("_"):  # internal ete3
                        continue
                    setattr(new_child, k, v)
                if not child.is_leaf():
                    clone_into(new_child, child)

        clone_into(comb_node, new_subtree)
        return global_tree
    def resolve_combs_iteratively(self, tree, ranked_features, abundance, max_passes=5):
        # Remember feature used at nodes to avoid immediate reuse
        used_feature_by_node = {}

        for _ in range(max_passes):
            combs = [n for n in tree.traverse() if len(n.children) >= 3]
            if not combs:
                break

            progressed = False
            for node in combs:
                # Labels present in this comb’s subtree
                labels_subset = [leaf.name for leaf in node.iter_leaves()]
                if len(labels_subset) < 3:
                    continue

                # Choose the next best feature not yet used on this node
                already = used_feature_by_node.get(node, set())
                candidate = None
                for f in ranked_features:
                    if f not in already:
                        candidate = f
                        break
                if candidate is None:
                    continue  # no remaining features

                # Rebuild subtree with that feature
                new_sub = self.rebuild_subtree_with_feature(labels_subset, candidate, abundance, root_name=node.name if hasattr(node, "name") else None)

                # If the rebuilt subtree still has a comby root (or identical topology), you can skip grafting
                comby = any(len(n.children) >= 3 for n in new_sub.traverse())
                if comby:
                    # still a comb; we can still graft (it may reduce depth elsewhere), but you may skip
                    pass

                # Graft
                self.graft_subtree(tree, node, new_sub)
                used_feature_by_node.setdefault(node, set()).add(candidate)
                progressed = True

            if not progressed:
                # no changes this pass -> stop
                break

        return tree

    
    def run_feature_based_clonaltree(self, dataset_name, fasta_file, useAbundance=True, revision=False, trim=False):
        # 1) Load & rank features
        features_result = self.load_features_for_dataset(dataset_name)
        if features_result is None:
            return None
        self.features_data, feature_names = features_result  # store for helpers
        ranked = self.rank_features_global(self.features_data, feature_names)
        top_feature = ranked[0]

        # 2) Read sequences
        labels, root, arraySeqs, abundance, dico = readFastaAbundance(fasta_file)
        
        # 3) Apply strict mode filtering - only keep sequences with features
        print("Applying strict mode filtering...")
        
        # Convert labels to abundance format for matching
        labels_with_abundance = []
        for label in labels:
            if label in abundance:
                abund = abundance[label]
                labels_with_abundance.append(f"{label}@{abund}")
            else:
                labels_with_abundance.append(label)
        
        # Filter to only sequences with features (strict mode)
        filtered_data = []
        for i, (lab, seq, ab) in enumerate(zip(labels_with_abundance, arraySeqs, abundance)):
            # Check if sequence has features (handle naive@1 = naive)
            has_features = False
            if lab in self.features_data:
                has_features = True
            elif lab == "naive@1" and "naive" in self.features_data:
                has_features = True
            
            if has_features:
                filtered_data.append((lab, seq, ab))
        
        if not filtered_data:
            print("❌ No sequences with features found!")
            return None
        
        filtered_labels, filtered_arraySeqs, filtered_abundance_list = zip(*filtered_data)
        
        # Convert abundance list back to dictionary using original labels
        filtered_abundance = {}
        for i, label in enumerate(filtered_labels):
            # Extract original label (remove @abundance part)
            if '@' in label:
                original_label = label.split('@')[0]
            else:
                original_label = label
            
            # Get abundance from original abundance dictionary
            if original_label in abundance:
                filtered_abundance[label] = abundance[original_label]
            else:
                filtered_abundance[label] = 1  # Default abundance
        
        print(f"   Kept {len(filtered_labels)} sequences with features (out of {len(labels)} total)")
        print(f"   Removed {len(labels) - len(filtered_labels)} sequences without features")

        # 4) Initial tree with top feature
        adjMatrix = self.create_single_feature_adjacency(filtered_labels, self.features_data, top_feature, normalize="rank")
        
        # Find root index in filtered data
        root_idx = 0
        if root in filtered_labels:
            root_idx = filtered_labels.index(root)
        elif f"{root}@{abundance.get(root, 1)}" in filtered_labels:
            root_idx = filtered_labels.index(f"{root}@{abundance.get(root, 1)}")
        
        tree, infoTree = primMST(adjMatrix, root_idx, filtered_labels, filtered_abundance, useAbundance)

        # 5) Iterative comb resolution with remaining features
        remaining = ranked[1:] if len(ranked) > 1 else []
        tree = self.resolve_combs_iteratively(tree, remaining, filtered_abundance, max_passes=5)

        # 6) Optional trim/revision
        if trim:
            tree = trimming(tree, filtered_labels, adjMatrix)
        if revision:
            tree = editTree(tree, adjMatrix, filtered_labels)

        infoTree = getDistances(tree)
        return tree, infoTree


def main():
    """Main function - test on dataset 54"""
    clonaltree = FeatureBasedClonalTree()
    
    # Test on dataset 54
    dataset_name = "dataset_54"
    fasta_file = "/Users/nikaabdollahi/Desktop/Reza_phylo_project/ClonalTree_code/aa_fasta_files/LLC_dataset54_1_200_aa_sequences.fa"
    
    if not os.path.exists(fasta_file):
        print(f"❌ Fasta file not found: {fasta_file}")
        return
    
    result = clonaltree.run_feature_based_clonaltree(
        dataset_name=dataset_name,
        fasta_file=fasta_file,
        useAbundance=True,
        revision=False,
        trim=False
    )
    
    if result:
        tree, infoTree = result
        print(f"\n🎉 FEATURE-BASED CLONALTREE COMPLETED")
        print(f"   Dataset: {dataset_name}")
        
        # Count comb nodes in final tree
        comb_nodes = [n for n in tree.traverse() if len(n.children) >= 3]
        print(f"   Final comb nodes: {len(comb_nodes)}")
        
        # Show tree structure
        print(f"   Tree structure:")
        print(tree.get_ascii(show_internal=True))
        
        # Show which features were used
        feature_names = list(clonaltree.features_data.values())[0].keys() if clonaltree.features_data else []
        ranked_features = clonaltree.rank_features_global(clonaltree.features_data, feature_names)
        print(f"   Top feature used: {ranked_features[0] if ranked_features else 'None'}")
        print(f"   All ranked features: {ranked_features[:5]}...")  # Show top 5
    else:
        print(f"❌ Failed to process {dataset_name}")

if __name__ == "__main__":
    main()
