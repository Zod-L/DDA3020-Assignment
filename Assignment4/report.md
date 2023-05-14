# Report-Assignment4
## Overview
In this assignment, I cluster data without using the labels. The data is first processed using PCA. K-means algorithm is then used to cluster data. I use two metrics: Silhouette Coefficient and Rand Index to evaluate the results.

## Data Processing Using PCA
Since the dataset is relatively small, with only 210 samples, in order to avoid overfitting, dimension reduction is needed. 
```python
# Load data
data = np.loadtxt("seeds_dataset.txt").transpose()
X = data[:7, :] # [7, N]
Y = data[7, :] # [N]
print(f"Number of samples: {data.shape[1]}")
```
As is shown in the above code block, in order to perform PCA dimension reduction, the data is arange in columns, with a shape of [D, N], where D is the dimension of each data, and N is the number of samples. Then PCA is performed by using the SVD decomposition provided by numpy package


```python
# Load data
def pca_reduce(X, K):
    # X: [D, N]
    _, N = X.shape
    miu = X.mean(axis=1, keepdims=True)
    cov = 1/N * (X - miu) @ (X - miu).transpose()
    U, S, _ = np.linalg.svd(cov)
    U = U[:, :K]

    return U.transpose() @ (X - miu), S
```
Mean(miu) and covariance matrix(cov) are calculated from the data. Then, SVD decomposition is perform on the covariance matrix, obtaining eigenvalues and eigenvectors. K eigenvectors are selected, According to the requirement, K=2. Actually, the eigenvalues also show that 2 eigenvectors are enough:
```python
Eigenvalues are:
 [1.07419301e+01 2.11931485e+00 7.32794138e-02 1.28261257e-02
 2.73513989e-03 1.56297146e-03 2.95142261e-05]

Subspace dimensions: 2
```
After reducing dimension into 2, data is normalized  into [0, 1] in order to assist training:
```python
Eigenvalues are:
min_val = X.min(1, keepdims=True)
max_val = X.max(1, keepdims=True)
X = (X - min_val) / (max_val - min_val)
Subspace dimensions: 2
```

## K-means Clustering
After processing the data, K-means algorithm is used to cluster this data. In short, K-means algorithm can be summarized as updating cluster assignment and centroid alternately. I implement the K-means algorithm using this strategy. The K-mean class consists of a initializer and 6 in-class methods.
```python
# K-means
class k_means:
    def __init__(self, init_centriods):
          ...

    def calc_dist(self, X):
          ...
    
    def infer(self, X):
          ...
    
    def fit(self, X):
          ...
        
    def silhouette_coefficient(self, X):
          ...
    
    def rand_idx(self, X, Y):
          ...

    def draw_fit(self):
          ...
```
The initializer takes a initializing centroids as the input. The initial centroids are randomly chosen from the data. According to the requirement, 3 centroids are selected. Besides, the show_colors attribute is used for visualization.
```python
# K-means
     def __init__(self, init_centriods):
          self.centroids = init_centriods.copy() # [K, C]
          self.K, self.C = self.centroids.shape
          self.show_colors = ["y", "g", "b"]
...
idx = random.sample(range(X.shape[1]), 3)
print(f"Use index {idx} samples to initialize centriods")
model = k_means(X[:, idx])
```
For inference, the distance between each sample and the 3 centroids are calculated, using calc_dist methods. The centroids and data X are tiled(repeated) to the same size, so the element-wise difference can be conducted conviniently. After difference operation, 2-norm is calculated along the features axis. The final output has a shape of [N, 3], where N is the number of samples, indicating the distance between each sample and the 3 centriods. After obtaining distance matrix, assignment can simply be obtained by taking the minimum along the 3-centroids axis.
```python
    def calc_dist(self, X):
        # X: [K, N]
        # centriods: [K, C]
        K, N = X.shape
        _, C =self.centroids.shape
        X = X[:, :, np.newaxis]
        X = np.tile(X, (1, 1, C)) # [K, N, C]
        centroids = np.tile(self.centroids[:, np.newaxis, :], (1, N, 1)) # [K, N, C]
        dist = np.linalg.norm(X - centroids, axis=0) # [N, C]
        return dist
    


    def infer(self, X):
        dist = self.calc_dist(X)
        return np.argmin(dist, axis=1) + 1
```

For fitting the model, with the help of the above 2 helper function, the process is quite easy. First, cluster assignment is obtained by calling the infer function. For centriods update, I choose to update each centroid separately, by taking the mean of all samples with the specific label. I use high-level indexing technique, as is showing in the following code.
```python
def fit(self, X):
        self.X = X
        prev_centroid = np.full(self.centroids.shape, -1)
        K, N = X.shape # [K, N]
        _, C = self.centroids.shape # [K, C]

        iter = 0
        while (np.any(prev_centroid != self.centroids)):
            # Copy last centroid
            prev_centroid = self.centroids.copy()


            # Update assignment
            r = self.infer(X) # [N]


            # Update centriods
            for i in range(0, C):
                idx = (r == (i+1))
                self.centroids[:, i] = X[:, idx].mean(axis=1)
            iter += 1



        print(f"After {iter} iterations, k-means converge")
```
Experiments show that model can converge with several iterations, usually less than 15. In one experiment, where the initial centroids are sample with index [27, 138, 14], the model converge with only 6 iterations. 
``` python
Use index [27, 138, 14] samples to initialize centriods
After 6 iterations, k-means converge
```
## Evaluation
I implement 2 metrics and use them to evaluate my model. Silhouette coefficient metric, simply saying, measure the performance of clustering model by checking whether each sample is much closer to the samples in the same cluster than the samples in the second nearest cluster. Therefore, for each sample, its distance between each sample in the same cluster and the second nearest should be calculated. I conduct this using 3 matrixs: dist_matrix, first_first and second_first. The i-j entry dist_matrix record the distance between the i-th sample and the j-th sample.The i-j entry of first_first matrix denote whether the i-th sample and the j-th sample are in the same cluster. The i-j entry of second_first matrix denote whether the second nearest cluster of the i-th sample is the first nearest cluster of the j-th. These matrix are created by crossing the meshgrid. For value $a$, i.e. the mean distance between a point and all other points in the same
cluster, the diagonal of the matrix are removed to prevent self-calculation. In theory, both $a$ and $b$ should be divided by 2 since the symmtric of the matrix. However, $a$ and $b$ are taking the division operation, which elimiate such scale. 
```python
    def silhouette_coefficient(self, X):
        # X: [K, N] 
        K,N = X.shape
        dist = self.calc_dist(X) #[N, C]
        first_idx = np.argmin(dist, axis=1) #[N]
        dist[np.arange(N), first_idx] = 1e10
        second_idx = np.argmin(dist, axis=1)
        tmp1, tmp2 = np.meshgrid(first_idx, first_idx)
        tmp3, _ = np.meshgrid(second_idx, second_idx)
        first_first = (tmp1 == tmp2).astype(np.int32)
        second_first = (tmp3 == tmp2).astype(np.int32)
        x1 = np.tile(X.transpose()[:, np.newaxis, :], (1, N, 1))
        x2 = np.tile(X.transpose()[np.newaxis, :, :], (N, 1, 1))
        dist_matrix = np.linalg.norm(x1 - x2, axis=-1)

         
        A = (first_first * dist_matrix).sum(axis=1) / (first_first.sum(axis=1) - 1)
        B = (second_first * dist_matrix).sum(axis=1) / (second_first.sum(axis=1))
        res = (B - A) / np.maximum(A, B)


        
        return res.mean()
```

For rand index metric, two predictions are used. One is the prediction given by my model, another is the groundtruth cluster of the dataset. In order to find the proportion of sample pairs that both in the same cluster of both not in, I apply the similar technique as silhouette coefficient, using crossing of meshgrid to get 2 matrixs pair and pred_pair, whose i-j entry indicate whether the i and j sample are in the same cluster, ommitting what cluster label exactly is.

```python
def rand_idx(self, X, Y):
        K, N = X.shape

        # The i-j entry means whether i,j has the same assignment
        tmp1, tmp2 = np.meshgrid(Y, Y)
        pair = (tmp1 == tmp2)
        pred_Y = self.infer(X)
        tmp1, tmp2 = np.meshgrid(pred_Y, pred_Y)
        pred_pair = (tmp1 == tmp2)

        a = (np.logical_and(pair, pred_pair).sum() - N) / 2
        b = np.logical_and(np.logical_not(pair), np.logical_not(pred_pair)).sum() / 2
        c = np.logical_and(pair, np.logical_not(pred_pair)).sum() / 2
        d = np.logical_and(np.logical_not(pair), pred_pair).sum() / 2
        return (a + b) / (a + b + c + d)
```

The model achieve results:
```
The silhouette coefficient of the model is 0.46046235809126435
The rand index of the model is 0.8833447254499887
```
Visualized as 
<div align=center>
<img src=output.png width="80%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 1: Visualize the final clustering result. Three colors means 3 different clusters, red X means the centroids </div>
</div><br><br>