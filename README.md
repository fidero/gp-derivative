# GP-derivative 
Code accompanying the article: [High-Dimensional Gaussian Process Inference with Derivatives](https://arxiv.org/abs/2102.07542) [to appear @ [ICML 2021](https://icml.cc/)]

## Summary
The kernel Gram matrix for gradient observations exhibits important structure. The figure below shows how the Gram matrix can be decomposed for usage with the matrix inversion lemma for high-dimensional inputs (D>N).
![RBF kernel with N=3 and D=10](fig/thumbnail.png "RBF kernel with N=3 and D=10.")

For high-dimensional input with few observations there can be a drastic speedup. 
![runtime comparison](fig/runtime.png "CPU comparison of Woodbury decomposition versus standard Cholesky for different D and N.")


## Citation
__In progress__

In the meantime you can cite the arXiv submission:
```
@misc{deroos2021highdimensional,
      title={High-Dimensional Gaussian Process Inference with Derivatives}, 
      author={Filip de Roos and Alexandra Gessner and Philipp Hennig},
      year={2021},
      eprint={2102.07542},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```