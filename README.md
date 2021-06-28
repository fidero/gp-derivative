# GP-derivative
Code accompanying the article: [High-Dimensional Gaussian Process Inference with Derivatives](https://arxiv.org/abs/2102.07542)

## Summary
The kernel Gram matrix for gradient observations exhibits important structure. The figure below shows how the Gram matrix can be decomposed for usage with the matrix inversion lemma for high-dimensional inputs ($D<N$).
![RBF kernel with $N=3$ and $D=10$](fig/thumbnail.png)
For high-dimensional input with few observations there can be a drastic speedup. 
![runtime comparison](fig/runtime.pdf)