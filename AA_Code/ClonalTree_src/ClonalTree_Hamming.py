import os
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
    Run the AA-Hamming ClonalTree algorithm.
    """

    print("Running AA-Hamming ClonalTree")
    print("Input FASTA:", inputFile)
    print("Output:", outputFile)
    print(
        "Settings → useAbundance:", useAbundance,
        "; revision:", revision,
        "; trim:", trim
    )

    # 1. Read AA sequences and abundance
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

    tree.write(format=1, outfile=outputFile)

    with open(outputFile + ".csv", "w") as f:
        f.write(infoTree)

    print("Done!")
    print("Tree saved to:", outputFile)
    print("CSV saved to:", outputFile + ".csv")

    return tree


if __name__ == "__main__":

    # AA_FASTA is located one level above ClonalTree_src
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "..", "New_AA", "AA_FASTA")

    input_dir = os.path.abspath(input_dir)

    for file in os.listdir(input_dir):

        if file.endswith(".fa"):

            input_path = os.path.join(input_dir, file)

            output_path = os.path.join(
                input_dir,
                file.replace(".fa", ".nk")
            )

            run_clonalTree_aa(
                inputFile=input_path,
                outputFile=output_path,
                useAbundance=True,
                revision=True,
                trim=True
            )

            print(f"Done: {file}")