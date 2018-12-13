# Neural Network Identifies Cell Types in scRNA-Seq

## Overview
NNICT (Neural Network Identifies Cell Types in scRNA-Seq) is a bioinformatic tool to quickly and accurately identify cell types in scRNA-Seq. For details, please read the paper:

## Prerequisite
python 3.6

python packages:
tensorflow 1.10+, numpy 1.14+, pandas 0.23+, argparse 1.1+, scipy 1.1+, scanpy 1.3+

## Convert format
We use HDF5 format for the scRNA-Seq expressiong matrix, which stores the compressed matrix and is fast to load. To convert the format, we first read the expression matrix as a pandas dataframe, then we use the to_hdf function to save the file as HDF5 format. For the to_hdf function, we use "dge", which stands for digital gene expression, for the key parameter.

### Usage
```
python nnict_format.py -i input_file -o output_prefix -f format
```

### Paramters
* -i	Path to the input file or the 10X directory
* -o	Prefix of the output file
* -f	Format of the input file (10X, txt, csv)

### Examples

#### Convert 10X format
```
python nnict_format.py -i ./data/pbmc_10x/GRCh38 -o pbmc_10x -f 10X
```

#### Convert txt format
```
python nnict_format.py -i ./data/pbmc.txt -o pbmc -f txt
```

## Predict cell types
We train a 4 layer (3 hidden layers) neural network on scRNA-Seq datasets with predifined cell types, then we use the trained parameters to predict cell types for other datasets.

### Usage
```
python nnict_predict.py -trs training_set -trl training_label -ts test_set -lr 0.0001 -ne 50 -ms 128 -pc True
```

### Parameters
* -trs	Path to the training set, must be HDF5 format with key "dge"
* -trl	Path to the training label (the cell types for the training set), must be txt format
* -ts	Path to test sets, must be HDF5 format with key "dge"
* -lr	Learning rate (default: 0.0001)
* -ne	Number of epochs (default: 50)
* -ms	Minibatch size (default: 128)
* -pc	Print cost (default: True)

### Example
```
python nnict_predict.py -trs ./pbmc_set.h5 -trl pbmc_label.txt.gz -ts test_set.h5 -lr 0.0001 -ne 50 -ms 128 -pc True
```
