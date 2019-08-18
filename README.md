# Automated Cell Type Identification using Neural Networks

## Overview
ACTINN (Automated Cell Type Identification using Neural Networks) is a bioinformatic tool to quickly and accurately identify cell types in scRNA-Seq. For details, please read the paper:
https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btz592/5540320
All datasets used in the paper are available here:
https://figshare.com/articles/ACTINN/8967116

## Prerequisite
python 3.6

python packages:
tensorflow 1.10+, numpy 1.14+, pandas 0.23+, argparse 1.1+, scipy 1.1+

## Convert format
We use HDF5 format for the scRNA-Seq expressiong matrix, which stores the compressed matrix and is fast to load. To convert the format, we first read the expression matrix as a pandas dataframe, then we use the to_hdf function to save the file as HDF5 format. For the to_hdf function, we use "dge", which stands for digital gene expression, for the key parameter.

### Usage
```
python actinn_format.py -i input_file -o output_prefix -f format
```

### Paramters
* -i	Path to the input file or the 10X directory
* -o	Prefix of the output file
* -f	Format of the input file (10X_V2, 10X_V3, txt, csv)

### Output
The output will be an HDF5 formated file named after the output prefix with ".h5" extension

### Examples

#### Convert 10X_V2 format
```
python actinn_format.py -i ./test_data/train_set_10x -o train_set -f 10X_V2
```

#### Convert 10X_V3 format
```
python actinn_format.py -i ./test_data/train_set_10x -o train_set -f 10X_V3
```

#### Convert txt format
```
python actinn_format.py -i ./test_data/train_set.txt.gz -o train_set -f txt
```

#### Convert csv format
```
python actinn_format.py -i ./test_data/train_set.csv.gz -o train_set -f csv
```

## Predict cell types
We train a 4 layer (3 hidden layers) neural network on scRNA-Seq datasets with predifined cell types, then we use the trained parameters to predict cell types for other datasets.

### Usage
```
python actinn_predict.py -trs training_set -trl training_label -ts test_set -lr learning_rat -ne num_epoch -ms minibatch_size -pc print_cost -op output_probability
```

### Parameters
* -trs	Path to the training set, must be HDF5 format with key "dge".
* -trl	Path to the training label (the cell types for the training set), must be tab separated text file with no column and row names.
* -ts	Path to test sets, must be HDF5 format with key "dge".
* -lr	Learning rate (default: 0.0001). We can increase the learning rate if the cost drops too slow, or decrease the learning rate if the cost drops super fast in the beginning and starts to fluctuate in later epochs.
* -ne	Number of epochs (default: 50). The number of epochs can be determined by looking at the cost after each epoch. If the cost starts to decrease very slowly after ceartain epoch, then the "ne" parameter should be set to that epoch number. 
* -ms	Minibatch size (default: 128). This parameter can be set larger when training a large dataset.
* -pc	Print cost (default: True). Whether to print cost after each 5 epochs.
* -op Output probabilities for each cell being the cell types in the training data (default: False).

### Output
The output will be a file named "predicted_label.txt". In the file, the first column will be the cell name, the second column will be the predicted cell type. 
If the "op" parameter is set to True, there will be another output file named "predicted_probablities.txt", where columns are cells and rows are cell types. The number in row i and column j will be the probablity that cell j being cell type i.

### Example
```
python actinn_predict.py -trs ./test_data/train_set.h5 -trl ./test_data/train_label.txt.gz -ts ./test_data/test_set.h5 -lr 0.0001 -ne 50 -ms 128 -pc True -op False
```

## Plots
We show an example on how to create a tSNE plot with the predicted cell types. The R command can be found in the "tSNE_Example" folder.
![tSNE Plot](https://github.com/mafeiyang/ACTINN/blob/master/tSNE_Example/tSNE_Plot.png)
