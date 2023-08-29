# Clustering
The goal of this project is to perform clustering on the  MNIST database  (or a sufficiently small subset of it) using:

    - mixture of Gaussians with diagonal covariance (Gaussian Naive Bayes with latent class label);

    - mean shift;

    - normalized cut.

The unsupervised classification must be performed at varying levels of dimensionality reduction through PCA  (say going from 2 to 200) in order to asses the effect of the dimensionality in accuracy and learning time.

Provide the code and the extracted clusters as the number of clusters k varies from 5 to 15, for the mixture of Gaussians and normalized-cut, while for mean shift vary the kernel width. For each value of k (or kernel width) provide the value of the Rand index:

$$
R=2(a+b)/(n(n-1))
$$

where:

    - n is the number of images in the dataset.

    - a is the number of pairs of images that represent the same digit and that are clustered together.
    
    - b is the number of pairs of images that represent different digits and that are placed in different clusters.