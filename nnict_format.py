import numpy as np
import pandas as pd
import scanpy.api as sc
import os
import argparse

def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to the input file or the 10X directory.")
    parser.add_argument("-o", "--output", type=str, help="Prefix of the output file.")
    parser.add_argument("-f", "--format", type=str, help="Format of the input file (10X, txt, csv).")
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.format == "10X":
        path = args.input
        if path[-1] != "/":
            path += "/"
        new = sc.read(path + 'matrix.mtx', cache=True).T  # transpose the data
        new.var_names = pd.read_csv(path + 'genes.tsv', header=None, sep='\t')[1]
        new.obs_names = pd.read_csv(path + 'barcodes.tsv', header=None)[0]
        new = pd.DataFrame(new.X.todense().transpose(), index=new.var_names, columns=new.obs_names)
        uniq_index = np.unique(new.index, return_index=True)[1]
        new = new.iloc[uniq_index,]
        new = new.loc[new.sum(axis=1)>0, :]
        print("Dimension of the matrix after removing non-zero rows:", new.shape)
        new.to_hdf(args.output+".h5", key="dge", mode="w", complevel=3)
        os.system("rm -rf cache")
    if args.format == "csv":
        new = pd.read_csv(args.input, index_col=0)
        uniq_index = np.unique(new.index, return_index=True)[1]
        new = new.iloc[uniq_index,]
        new = new.loc[new.sum(axis=1)>0, :]
        print("Dimension of the matrix after removing non-zero rows:", new.shape)
        new.to_hdf(args.output+".h5", key="dge", mode="w", complevel=3)
    if args.format == "txt":
        new = pd.read_csv(args.input, index_col=0, sep="\t")
        uniq_index = np.unique(new.index, return_index=True)[1]
        new = new.iloc[uniq_index,]
        new = new.loc[new.sum(axis=1)>0, :]
        print("Dimension of the matrix after removing non-zero rows:", new.shape)
        new.to_hdf(args.output+".h5", key="dge", mode="w", complevel=3)
