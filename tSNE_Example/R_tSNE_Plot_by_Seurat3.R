library(Seurat)
library(dplyr)
library(rhdf5)
library(Matrix)

# Read in the data
h5_file = h5read("./mouse_leukocyte.h5", name="dge")
sample.data = t(h5_file$block0_values)
colnames(sample.data) = h5_file$axis0
rownames(sample.data) = h5_file$axis1
sample.data = as.data.frame(sample.data)

# Make seurat object
sample = CreateSeuratObject(counts=sample.data, min.cells=1, min.features=100)

# Normalize data the data
sample = NormalizeData(object = sample, normalization.method = "LogNormalize",
                       scale.factor = 10000)

# Find variable Genes and scale the data
sample = FindVariableFeatures(object=sample, selection.method="vst")
sample = ScaleData(object=sample)

# Run PCA
sample = RunPCA(object=sample, verbose=FALSE)

# Find clusters and reduce dimensions 
pca_dims = 1:10
sample = RunTSNE(object=sample, dims=pca_dims, verbose=FALSE)
sample = FindNeighbors(object=sample, dims=pca_dims, verbose=FALSE)
sample = FindClusters(object=sample, resolution=0.5, verbose=FALSE)

# Plot use the predicted label
label = read.table("./predicted_label.txt", sep="\t", header=TRUE, row.names=1)
sample$label = label[names(Idents(sample)),1]
DimPlot(sample, label=FALSE, reduction="tsne", group.by="label", pt.size=1)
