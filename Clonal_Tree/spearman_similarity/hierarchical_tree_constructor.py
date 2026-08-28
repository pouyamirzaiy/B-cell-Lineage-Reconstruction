#!/usr/bin/env python3
#-*-coding:Utf-8-*-

"""
Hierarchical Feature-Based Tree Constructor
Implements the iterative, feature-hierarchical strategy for phylogenetic tree construction
to address tied/comb topology issues in multi-feature similarity matrices.

Algorithm:
1. Extract structural/physico-chemical features from IGH amino-acid models
2. Select top 5 most consistent features across datasets
3. Build similarity matrices using these features (Spearman correlation)
4. Convert similarity → distance and build trees
5. Address comb topologies with iterative, feature-hierarchical resolution:
   - Start with most informative feature (highest CV, many unique values)
   - For unresolved subtrees, use next most informative feature
   - Recursively resolve until whole tree is resolved
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

class HierarchicalTreeConstructor:
    """
    Implements hierarchical feature-based tree construction to resolve comb topologies.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the constructor with data path.
        
        Args:
            data_path: Path to the extracted Sarah Kaveh materials
        """
        self.data_path = Path(data_path)
        self.features = {}
        self.similarity_matrices = {}
        self.feature_informativeness = {}
        self.trees = {}
    
    def similarity_to_distance(self, similarity_values):
        """
        Convert similarity to distance using robust normalization.
        
        Formula: N = (similarity - min_similarity) / (max_similarity - min_similarity)
                Distance = 1 - N
        
        Args:
            similarity_values: Array or matrix of similarity values
            
        Returns:
            Array or matrix of distance values in [0,1] range
        """
        similarity_array = np.array(similarity_values)
        min_sim = np.min(similarity_array)
        max_sim = np.max(similarity_array)
        
        # Handle edge case where all similarities are identical
        if max_sim == min_sim:
            return np.zeros_like(similarity_array)
        
        # Normalize to [0,1] then convert to distance
        N = (similarity_array - min_sim) / (max_sim - min_sim)
        distance = 1 - N
        
        return distance
    
    def prim_mst(self, distance_matrix: np.ndarray, proteins: List[str]) -> nx.Graph:
        """
        Build tree using Prim's MST algorithm (same as original ClonalTree).
        
        Args:
            distance_matrix: Distance matrix between proteins
            proteins: List of protein names
            
        Returns:
            NetworkX graph representing the MST tree
        """
        n_proteins = len(proteins)
        INF = float('inf')
        
        # Create adjacency matrix with INF on diagonal
        adj_matrix = distance_matrix.copy()
        np.fill_diagonal(adj_matrix, INF)
        
        # Initialize tree
        tree = nx.Graph()
        tree.add_node(proteins[0])  # Start with first protein as root
        
        visited_nodes = [0]  # Track visited node indices
        
        while len(visited_nodes) < n_proteins:
            # Find minimum edge from visited to unvisited nodes
            min_distance = INF
            min_i, min_j = -1, -1
            
            for i in visited_nodes:
                for j in range(n_proteins):
                    if j not in visited_nodes and adj_matrix[i][j] < min_distance:
                        min_distance = adj_matrix[i][j]
                        min_i, min_j = i, j
            
            if min_i == -1 or min_j == -1:
                break  # No more edges to add
            
            # Add edge to tree
            tree.add_edge(proteins[min_i], proteins[min_j], weight=min_distance)
            
            # Mark node as visited
            visited_nodes.append(min_j)
            
            # Remove edge from consideration
            adj_matrix[min_i][min_j] = INF
            adj_matrix[min_j][min_i] = INF
        
        return tree
    
    def load_per_feature_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load per-feature Spearman similarity matrices.
        
        Returns:
            Dictionary mapping feature names to similarity matrices
        """
        print("Loading per-feature Spearman similarity matrices...")
        
        per_feature_path = self.data_path / "per_feature_spearman"
        if not per_feature_path.exists():
            raise FileNotFoundError(f"Per-feature data not found at {per_feature_path}")
        
        features = {}
        
        # Look for dataset folders (01, 02, etc.)
        for dataset_folder in sorted(per_feature_path.glob("[0-9][0-9]")):
            dataset_name = dataset_folder.name
            print(f"  Processing dataset {dataset_name}...")
            
            # Look for feature matrices in pairs/ or matrices/ subfolders
            pairs_path = dataset_folder / "pairs"
            matrices_path = dataset_folder / "matrices"
            
            if pairs_path.exists():
                # Load pairs files
                for pairs_file in pairs_path.glob("*_pairs.csv"):
                    feature_name = pairs_file.stem.replace("_pairs", "")
                    if feature_name not in features:
                        features[feature_name] = {}
                    
                    try:
                        pairs_df = pd.read_csv(pairs_file)
                        if 'Protein1' in pairs_df.columns and 'Protein2' in pairs_df.columns:
                            features[feature_name][dataset_name] = pairs_df
                    except Exception as e:
                        print(f"    Warning: Could not load {pairs_file}: {e}")
            
            elif matrices_path.exists():
                # Load matrix files
                for matrix_file in matrices_path.glob("*_matrix.csv"):
                    feature_name = matrix_file.stem.replace("_matrix", "")
                    if feature_name not in features:
                        features[feature_name] = {}
                    
                    try:
                        matrix_df = pd.read_csv(matrix_file, index_col=0)
                        features[feature_name][dataset_name] = matrix_df
                    except Exception as e:
                        print(f"    Warning: Could not load {matrix_file}: {e}")
        
        self.features = features
        print(f"Loaded {len(features)} features across datasets")
        return features
    
    def calculate_feature_informativeness(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate informativeness metrics for each feature across datasets.
        
        Returns:
            Dictionary with informativeness metrics for each feature
        """
        print("Calculating feature informativeness...")
        
        informativeness = {}
        
        for feature_name, datasets in self.features.items():
            print(f"  Analyzing feature: {feature_name}")
            
            cv_scores = []
            unique_value_counts = []
            identical_value_counts = []
            
            for dataset_name, data in datasets.items():
                if isinstance(data, pd.DataFrame):
                    if 'Protein1' in data.columns and 'Protein2' in data.columns:
                        # Pairs format - convert to matrix
                        similarity_values = data['Spearman'].values if 'Spearman' in data.columns else data.iloc[:, 2].values
                    else:
                        # Matrix format
                        similarity_values = data.values.flatten()
                    
                    # Calculate metrics
                    cv = np.std(similarity_values) / np.mean(similarity_values) if np.mean(similarity_values) != 0 else 0
                    unique_count = len(np.unique(similarity_values))
                    identical_count = len(similarity_values) - unique_count
                    
                    cv_scores.append(cv)
                    unique_value_counts.append(unique_count)
                    identical_value_counts.append(identical_count)
            
            # Aggregate metrics across datasets
            informativeness[feature_name] = {
                'mean_cv': np.mean(cv_scores) if cv_scores else 0,
                'mean_unique_values': np.mean(unique_value_counts) if unique_value_counts else 0,
                'mean_identical_values': np.mean(identical_value_counts) if identical_value_counts else 0,
                'informativeness_score': np.mean(cv_scores) * np.mean(unique_value_counts) / (np.mean(identical_value_counts) + 1) if cv_scores else 0
            }
        
        self.feature_informativeness = informativeness
        return informativeness
    
    def select_top_features(self, n_features: int = 5) -> List[str]:
        """
        Select the top N most informative features.
        
        Args:
            n_features: Number of top features to select
            
        Returns:
            List of top feature names ordered by informativeness
        """
        if not self.feature_informativeness:
            self.calculate_feature_informativeness()
        
        # Sort features by informativeness score
        sorted_features = sorted(
            self.feature_informativeness.items(),
            key=lambda x: x[1]['informativeness_score'],
            reverse=True
        )
        
        top_features = [feature[0] for feature in sorted_features[:n_features]]
        
        print(f"Selected top {n_features} features:")
        for i, feature in enumerate(top_features, 1):
            score = self.feature_informativeness[feature]['informativeness_score']
            print(f"  {i}. {feature}: {score:.4f}")
        
        return top_features
    
    def build_initial_tree(self, dataset_name: str, feature_name: str) -> nx.Graph:
        """
        Build initial tree using the most informative feature.
        
        Args:
            dataset_name: Name of the dataset
            feature_name: Name of the feature to use
            
        Returns:
            NetworkX graph representing the tree
        """
        print(f"Building initial tree for dataset {dataset_name} using feature {feature_name}")
        
        if feature_name not in self.features or dataset_name not in self.features[feature_name]:
            raise ValueError(f"Feature {feature_name} or dataset {dataset_name} not found")
        
        data = self.features[feature_name][dataset_name]
        
        # Convert to distance matrix
        if 'Protein1' in data.columns and 'Protein2' in data.columns:
            # Pairs format
            similarity_col = 'Spearman' if 'Spearman' in data.columns else data.columns[2]
            proteins = list(set(data['Protein1'].tolist() + data['Protein2'].tolist()))
            
            # Create similarity matrix
            n_proteins = len(proteins)
            protein_to_idx = {p: i for i, p in enumerate(proteins)}
            
            similarity_matrix = np.ones((n_proteins, n_proteins))
            np.fill_diagonal(similarity_matrix, 1.0)
            
            for _, row in data.iterrows():
                i = protein_to_idx[row['Protein1']]
                j = protein_to_idx[row['Protein2']]
                similarity_matrix[i, j] = row[similarity_col]
                similarity_matrix[j, i] = row[similarity_col]
        else:
            # Matrix format
            similarity_matrix = data.values
            proteins = data.index.tolist()
        
        # Convert similarity to distance using robust normalization
        distance_matrix = self.similarity_to_distance(similarity_matrix)
        
        # Build tree using Prim's MST (same as original ClonalTree)
        tree = self.prim_mst(distance_matrix, proteins)
        
        return tree
    
    def identify_comb_nodes(self, tree: nx.Graph, threshold: float = 0.1) -> List[str]:
        """
        Identify comb nodes in MST tree (nodes with many similar-weight edges).
        
        Args:
            tree: NetworkX graph representing the MST tree
            threshold: Distance threshold for considering edges as similar
            
        Returns:
            List of comb node identifiers
        """
        comb_nodes = []
        
        for node in tree.nodes():
            neighbors = list(tree.neighbors(node))
            if len(neighbors) > 2:  # More than 2 connections indicates potential comb
                edges = [(node, neighbor) for neighbor in neighbors]
                weights = [tree.edges[edge]['weight'] for edge in edges]
                
                # Check if edge weights are similar (indicating comb topology)
                if len(weights) > 2 and np.std(weights) < threshold:
                    comb_nodes.append(node)
        
        return comb_nodes
    
    def resolve_comb_node(self, tree: nx.Graph, comb_node: str, feature_name: str, 
                         dataset_name: str) -> nx.Graph:
        """
        Resolve a comb node using the next most informative feature.
        
        Args:
            tree: NetworkX graph representing the tree
            comb_node: Identifier of the comb node to resolve
            feature_name: Feature to use for resolution
            dataset_name: Name of the dataset
            
        Returns:
            Updated NetworkX graph
        """
        print(f"Resolving comb node {comb_node} using feature {feature_name}")
        
        # Get proteins in the subtree rooted at comb_node
        subtree_proteins = self._get_subtree_proteins(tree, comb_node)
        
        if len(subtree_proteins) < 3:
            return tree  # Can't resolve with fewer than 3 proteins
        
        # Get similarity data for these proteins using the specified feature
        if feature_name not in self.features or dataset_name not in self.features[feature_name]:
            print(f"Warning: Feature {feature_name} not available for dataset {dataset_name}")
            return tree
        
        data = self.features[feature_name][dataset_name]
        
        # Filter data to only include proteins in the subtree
        if 'Protein1' in data.columns and 'Protein2' in data.columns:
            subtree_data = data[
                (data['Protein1'].isin(subtree_proteins)) & 
                (data['Protein2'].isin(subtree_proteins))
            ].copy()
        else:
            # Matrix format - filter rows and columns
            subtree_data = data.loc[subtree_proteins, subtree_proteins]
        
        if len(subtree_data) == 0:
            print(f"Warning: No data available for subtree proteins")
            return tree
        
        # Build new subtree using the specified feature
        new_subtree = self._build_subtree(subtree_data, subtree_proteins)
        
        # Replace the comb node with the new subtree
        updated_tree = self._replace_subtree(tree, comb_node, new_subtree)
        
        return updated_tree
    
    def _get_subtree_proteins(self, tree: nx.Graph, root_node: str) -> Set[str]:
        """Get all leaf proteins in the subtree rooted at root_node."""
        proteins = set()
        
        def dfs(node, visited):
            if node in visited:
                return
            visited.add(node)
            
            if tree.nodes[node].get('type') == 'leaf':
                proteins.add(node)
            else:
                for neighbor in tree.neighbors(node):
                    if neighbor not in visited:
                        dfs(neighbor, visited)
        
        dfs(root_node, set())
        return proteins
    
    def _build_subtree(self, data: pd.DataFrame, proteins: List[str]) -> nx.Graph:
        """Build a subtree from similarity data."""
        # Implementation similar to build_initial_tree but for subtree
        # This is a simplified version - you may want to implement more sophisticated subtree building
        subtree = nx.Graph()
        
        for protein in proteins:
            subtree.add_node(protein, type='leaf')
        
        # Simple star topology for now - can be improved
        if len(proteins) > 1:
            center = proteins[0]
            for protein in proteins[1:]:
                subtree.add_edge(center, protein, weight=0.5)
        
        return subtree
    
    def _replace_subtree(self, tree: nx.Graph, old_root: str, new_subtree: nx.Graph) -> nx.Graph:
        """Replace a subtree in the tree with a new subtree."""
        # Remove old subtree
        subtree_nodes = list(nx.descendants(tree, old_root)) + [old_root]
        tree.remove_nodes_from(subtree_nodes)
        
        # Add new subtree
        tree.add_nodes_from(new_subtree.nodes(data=True))
        tree.add_edges_from(new_subtree.edges(data=True))
        
        return tree
    
    def hierarchical_tree_construction(self, dataset_name: str, top_features: List[str]) -> nx.Graph:
        """
        Perform hierarchical tree construction using multiple features.
        
        Args:
            dataset_name: Name of the dataset
            top_features: List of features ordered by informativeness
            
        Returns:
            Final resolved tree
        """
        print(f"Starting hierarchical tree construction for dataset {dataset_name}")
        
        if not top_features:
            raise ValueError("No features provided")
        
        # Start with the most informative feature
        primary_feature = top_features[0]
        tree = self.build_initial_tree(dataset_name, primary_feature)
        
        # Iteratively resolve comb nodes using remaining features
        for i, feature in enumerate(top_features[1:], 1):
            print(f"  Iteration {i}: Using feature {feature}")
            
            comb_nodes = self.identify_comb_nodes(tree)
            if not comb_nodes:
                print(f"  No comb nodes found - tree is fully resolved!")
                break
            
            print(f"  Found {len(comb_nodes)} comb nodes to resolve")
            
            for comb_node in comb_nodes:
                tree = self.resolve_comb_node(tree, comb_node, feature, dataset_name)
        
        # Final check for remaining comb nodes
        final_comb_nodes = self.identify_comb_nodes(tree)
        if final_comb_nodes:
            print(f"Warning: {len(final_comb_nodes)} comb nodes remain unresolved")
        else:
            print("Tree construction completed - all comb nodes resolved!")
        
        return tree
    
    def visualize_tree(self, tree: nx.Graph, dataset_name: str, output_path: str = None):
        """
        Visualize the constructed tree.
        
        Args:
            tree: NetworkX graph representing the tree
            dataset_name: Name of the dataset
            output_path: Path to save the visualization
        """
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(tree, k=1, iterations=50)
        
        # Draw nodes
        leaf_nodes = [n for n in tree.nodes() if tree.nodes[n].get('type') == 'leaf']
        internal_nodes = [n for n in tree.nodes() if tree.nodes[n].get('type') == 'internal']
        
        nx.draw_networkx_nodes(tree, pos, nodelist=leaf_nodes, 
                              node_color='lightblue', node_size=100, alpha=0.7)
        nx.draw_networkx_nodes(tree, pos, nodelist=internal_nodes, 
                              node_color='red', node_size=50, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(tree, pos, alpha=0.5)
        
        # Draw labels for leaf nodes only
        leaf_labels = {n: n for n in leaf_nodes}
        nx.draw_networkx_labels(tree, pos, leaf_labels, font_size=8)
        
        plt.title(f"Hierarchical Tree for Dataset {dataset_name}")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Tree visualization saved to {output_path}")
        
        plt.show()

def main():
    """Example usage of the HierarchicalTreeConstructor."""
    
    # Initialize constructor
    data_path = "/"
    constructor = HierarchicalTreeConstructor(data_path)
    
    # Load data (after RAR files are extracted)
    try:
        features = constructor.load_per_feature_data()
        
        # Calculate feature informativeness
        informativeness = constructor.calculate_feature_informativeness()
        
        # Select top 5 features
        top_features = constructor.select_top_features(n_features=5)
        
        # Build hierarchical tree for a sample dataset
        sample_dataset = "01"  # Use first available dataset
        if any(sample_dataset in datasets for datasets in features.values()):
            tree = constructor.hierarchical_tree_construction(sample_dataset, top_features)
            
            # Visualize the result
            constructor.visualize_tree(tree, sample_dataset, 
                                     f"results/trees/hierarchical_tree_{sample_dataset}.png")
        else:
            print("No data available for sample dataset")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please extract the RAR files first using the extract_sarah_materials.py script")

if __name__ == "__main__":
    main()
