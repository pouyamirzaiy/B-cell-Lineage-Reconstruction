import numpy as np
import pandas as pd
from optparse import OptionParser
from MSTree import *
from BasicTree import *
import sys 
import os

#---------------------------------------------------------------------------
def makeBoolean(var):
    if var == '1':
        return True
    else:
        return False

#---------------------------------------------------------------------------
def readSpearmanMatrix(csvFile):
    """
    Read spearman similarity matrix from CSV and convert to distance matrix
    Also extract abundance information from sequence names
    """
    # Read the CSV file
    df = pd.read_csv(csvFile, index_col=0)
    
    # Get sequence names and convert to list
    labels = df.index.tolist()
    
    # Find the root (naive sequence)
    root = 0
    for i, label in enumerate(labels):
        if label == 'naive':
            root = i
            break
    
    # Extract abundance information from sequence names
    abundance = {}
    for label in labels:
        if '@' in label:
            seq_name, abund = label.split('@')
            abundance[label] = int(abund)
        else:
            abundance[label] = 1
    
    # Convert similarity matrix to distance matrix
    # Option 1: Simple conversion (1 - similarity) with scaling
    similarity_matrix = df.values
    distance_matrix = 1 - similarity_matrix
    
    # Scale distances to a reasonable range (multiply by 100 to get 0-5 range)
    # This prevents the negative distance issue in updateDistances function
    distance_matrix = distance_matrix * 100
    
    # Option 2: Alternative conversion using negative log (more mathematically sound)
    # Uncomment the next 3 lines and comment out the scaling above to use this method
    # distance_matrix = -np.log(similarity_matrix + 1e-10)  # Add small epsilon to avoid log(0)
    # # Scale to reasonable range
    # distance_matrix = distance_matrix * 10
    
    # Set diagonal to 0 (no self-distance)
    np.fill_diagonal(distance_matrix, 0)
    
    return labels, root, distance_matrix, abundance

#---------------------------------------------------------------------------
def createAdjMatrixFromDistance(distance_matrix):
    """
    Create adjacency matrix from distance matrix
    """
    return distance_matrix

#===================================================================================
#                        Main
#===================================================================================
def main():
    usage = "python clonalTree_spearman.py -i <spearman_csv_file> -o <outputFile> \n"
    parser = OptionParser(usage)
    parser.add_option("-i", "--spearmanFile", dest="spearmanFile", help="spearman similarity matrix in CSV format")
    parser.add_option("-o", "--outputFile", dest="outputFile", help="output file")
    parser.add_option("-a", "--useAbundance", dest="useAbundance", help="if 1 it uses abundance")
    parser.add_option("-r", "--revision", dest="revision", help="if 1 it performs revision")
    parser.add_option("-t", "--trim", dest="trim", help="if 1 it performs trimming tree")
    
    (options, args) = parser.parse_args()
    if len(sys.argv) < 5:
        parser.error("incorrect number of arguments")
    
    spearmanFile = options.spearmanFile
    outputFile = options.outputFile
    useAbundance = options.useAbundance
    revision = options.revision
    trim = options.trim

    useAbundance = makeBoolean(useAbundance)
    revision = makeBoolean(revision)
    trim = makeBoolean(trim)
    
    print("Parameter setting = useAbundance: ", useAbundance, "; revision: ", revision, "; trim:", trim)
    print("Reading spearman similarity matrix from:", spearmanFile)

    # Read spearman matrix and convert to distance
    labels, root, distance_matrix, abundance = readSpearmanMatrix(spearmanFile)
    
    print(f"Found {len(labels)} sequences")
    print(f"Root sequence: {labels[root]}")
    print(f"Abundance range: {min(abundance.values())} to {max(abundance.values())}")
    
    # Create adjacency matrix from distance matrix
    adjMatrix = createAdjMatrixFromDistance(distance_matrix)
    
    print("Building tree using MST algorithm...")
    
    # Build tree using MST
    tree, infoTree = primMST(adjMatrix, root, labels, abundance, useAbundance)
    
    if trim:
        print("Performing tree trimming...")
        tree = trimming(tree, labels, adjMatrix)
    
    if revision:
        print("Performing tree revision...")
        tree = editTree(tree, adjMatrix, labels)
    
    # Get final distances
    infoTree = getDistances(tree)
    
    # Write output
    tree.write(format=1, outfile=outputFile)
    f = open(outputFile + '.csv', 'w')
    f.write(infoTree)
    f.close()
    
    print('Done! Output files:')
    print(f'  Tree: {outputFile}')
    print(f'  Info: {outputFile}.csv')

#===================================================================================
if __name__ == "__main__":
    main()
