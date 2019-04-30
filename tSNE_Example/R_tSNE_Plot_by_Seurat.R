library(Seurat)
library(dplyr)
library(rhdf5)

# Read in the data
h5_file = h5read("./mouse_leukocyte.h5", name="dge")
sample.data = t(h5_file$block0_values)
colnames(sample.data) = h5_file$axis0
rownames(sample.data) = h5_file$axis1

# Make seurat object
sample <- CreateSeuratObject(raw.data = sample.data)

# Normalize data the data
sample <- NormalizeData(object = sample, normalization.method = "LogNormalize", 
                        scale.factor = 10000)

# Find variable Genes and scale the data
sample <- FindVariableGenes(object = sample)
sample <- ScaleData(object = sample)

# Run PCA
sample <- RunPCA(object = sample, pc.genes = sample@var.genes, do.print = FALSE)

# Find clusters and Run t-SNE
sample <- FindClusters(object = sample, print.output = 0)
sample <- RunTSNE(object = sample, do.fast = TRUE)

# Plot use the predicted label
label = read.table("./predicted_label.txt", sep="\t", header=TRUE, row.names=1)
sample@meta.data$label = label[names(sample@ident),1]
TSNEPlot(object=sample, do.label=FALSE, pt.size=1, group.by="label")
