# B-cell Lineage Reconstruction (ClonalTree-Based Analysis)

## Project Status

**Important Notice:**  
This repository documents ongoing research on alternative approaches for B-cell lineage reconstruction using the ClonalTree framework.

The project is currently under active development and has not yet been completed. Some implementation details and experimental results are intentionally omitted because the work is ongoing and is being prepared for future publication.

This repository describes the motivation, methodology, and current progress of the research.

---

## 1. Introduction

B-cell lineage reconstruction is an important problem in computational immunology. During affinity maturation, B-cells accumulate mutations that generate diverse populations of related antibody sequences. Reconstructing the evolutionary relationships among these sequences provides insight into immune responses, clonal evolution, and antibody development.

Many existing lineage reconstruction methods rely primarily on nucleotide-level sequence similarity. ClonalTree is one such method that combines sequence distances, abundance information, and Minimum Spanning Tree (MST) construction to efficiently infer lineage relationships.

While nucleotide-based approaches are computationally efficient, they may not always capture biologically meaningful relationships between sequences. Mutations that appear significant at the nucleotide level may have little or no effect at the protein level, while biologically important amino-acid substitutions may not be adequately represented by simple sequence distance measures.

The objective of this research is to investigate whether alternative sequence representations can improve lineage reconstruction while maintaining the computational efficiency of ClonalTree. In particular, we explore:

- Traditional nucleotide-based lineage reconstruction  
- Amino-acid based lineage reconstruction using protein-level sequence similarity  
- Biologically informed reconstruction methods incorporating amino-acid physicochemical properties  

---

## 2. Amino-Acid Based Reconstruction

### 2.1 Motivation

Traditional ClonalTree reconstruction relies on nucleotide-level Hamming distance. Although computationally simple and effective, nucleotide similarity does not always reflect functional similarity between proteins.

To investigate whether protein-level information provides a more biologically meaningful representation of clonal evolution, an amino-acid based version of the ClonalTree pipeline was developed.

---

### 2.2 Method

The LLC benchmark datasets provide aligned nucleotide sequences but do not directly provide amino-acid sequences suitable for analysis.

Amino-acid sequences were obtained using the IMGT framework and used as input for reconstruction.

Workflow:

1. Obtain amino-acid sequences from IMGT  
2. Compute pairwise amino-acid Hamming distances  
3. Construct amino-acid distance matrix  
4. Apply ClonalTree MST procedure  
5. Apply trimming and revision steps when required  
6. Export lineage tree  

---

### 2.3 Example Output

Figure 1: Amino-acid based lineage tree generated using ClonalTree.

---

## 3. Feature-Based Reconstruction

### 3.1 Initial Implementation

In the first approach, each sequence was represented using a single numerical feature extracted from a feature table. Distances were computed as:

d(i, j) = |f(i) − f(j)|

This method produced valid lineage trees but often resulted in chain-like structures due to the low dimensionality of the feature representation.

---

### 3.2 Improved Feature-Based Reconstruction

To address this limitation, a multidimensional feature representation was introduced:

Xi = (f1(i), f2(i), ..., fk(i))

Distances are computed using Euclidean distance:

d(i, j) = sqrt( Σ (Xi,m − Xj,m)² )

Improvements include:

- Robust mapping between FASTA and feature tables  
- Handling missing feature values  
- Consistent abundance labeling  
- More stable MST construction  

This significantly improves tree topology and biological interpretability.

---

### 3.3 Example Outputs

Figure 2: Initial single-feature reconstruction  
Figure 3: Improved multidimensional feature-based reconstruction (Dataset 01)  
Figure 4: Improved multidimensional feature-based reconstruction (Dataset 02)

---

## 4. Biologically Informed Reconstruction (Ongoing Work)

This stage focuses on developing similarity measures that incorporate biochemical and physicochemical properties of amino acids.

Unlike standard Hamming distance, this approach considers that different amino-acid substitutions may have different biological impacts.

The goal is to improve lineage reconstruction accuracy while preserving computational efficiency.

This part of the project is still under active development and is not fully included in this repository.

---

## 5. Current Status

The project currently includes:

- Nucleotide-based ClonalTree reconstruction  
- Amino-acid based reconstruction  
- Feature-based reconstruction (single and multidimensional)  
- Development of biologically informed similarity metrics  
- Benchmark evaluation on LLC datasets  

---

## 6. Future Work

Future work will focus on:

- Comprehensive comparison of reconstruction methods  
- Benchmark evaluation and validation  
- Improved biologically informed distance metrics  
- Preparation for publication  

---

## Author

Pouya Mirzaeizadeh

---

## Notes

This repository is part of ongoing research. Some details and results are intentionally omitted until publication.
