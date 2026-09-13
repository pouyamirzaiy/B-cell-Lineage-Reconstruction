import sys
from pathlib import Path

# Find repository root
current = Path(__file__).resolve()

for path in [current.parent, *current.parents]:
    if (path / "AA_Code").is_dir():
        repo_root = path
        break
else:
    raise FileNotFoundError("Could not find the repository root.")

# Add ClonalTree source directory to Python path
clonaltree_src = repo_root / "AA_Code" / "ClonalTree_src"
sys.path.insert(0, str(clonaltree_src))

import numpy as np
from MSTree import primMST
from BasicTree import trimming, editTree, getDistances
from BasicSeq import readFastaAbundance, createAdjMatrix


def run_clonalTree_aa(
    inputFile,
    outputFile,
    useAbundance=True,
    revision=True,
    trim=True
):
    """
    Run AA-Hamming ClonalTree.
    """

    print("Running AA-Hamming ClonalTree")
    print("Input FASTA:", inputFile)
    print("Output:", outputFile)
    print(
        "Settings → useAbundance:", useAbundance,
        "; revision:", revision,
        "; trim:", trim
    )

    # 1. Read AA sequences + abundance
    labels, root, arraySeqs, abundance, _ = readFastaAbundance(inputFile)

    print(f"Found {len(labels)} sequences")
    print(f"Root sequence: {labels[root]}")
    print(
        f"Abundance range: "
        f"{min(abundance.values())} to {max(abundance.values())}"
    )

    # 2. Create adjacency matrix using AA Hamming distance
    adjMatrix = createAdjMatrix(arraySeqs)

    print("Building MST tree...")

    # 3. Build MST tree
    tree, infoTree = primMST(
        adjMatrix,
        root,
        labels,
        abundance,
        useAbundance
    )

    # 4. Optional trimming and revision
    if trim:
        print("Trimming tree...")
        tree = trimming(tree, labels, adjMatrix)

    if revision:
        print("Revising tree...")
        tree = editTree(tree, adjMatrix, labels)

    # 5. Export results
    infoTree = getDistances(tree)

    outputFile = Path(outputFile)

    tree.write(format=1, outfile=str(outputFile))

    with open(str(outputFile) + ".csv", "w") as f:
        f.write(infoTree)

    print("Done!")
    print("Tree saved to:", outputFile)
    print("CSV saved to:", str(outputFile) + ".csv")

    return tree


def main():

    base_dir = repo_root / "AA_Code" / "ground_truth"

    for i in range(1, 5):

        folder_name = f"asestra_eco_{i}"
        folder = base_dir / folder_name

        input_path = folder / "AA_sequences.fasta"
        output_path = folder / "AA_sequences.nk"

        if not input_path.exists():
            print(
                f"Skipping folder {folder_name}: "
                f"{input_path} not found."
            )
            continue

        run_clonalTree_aa(
            inputFile=input_path,
            outputFile=output_path,
            useAbundance=True,
            revision=True,
            trim=True
        )

        print(f"Done: {folder_name}")


if __name__ == "__main__":
    main()