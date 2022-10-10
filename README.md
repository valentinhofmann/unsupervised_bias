# Unsupervised Detection of Ideological Bias in Contextualized Embeddings

This repository contains the code for the ICML 2022 paper [Unsupervised Detection of Contextualized Embedding
Bias with Application to Ideology](https://proceedings.mlr.press/v162/hofmann22a.html).

# Dependencies

The code requires `Python>=3.6`, `numpy>=1.18`, `torch>=1.2`, and `torch_geometric>=1.6`.

# Usage

To replicate the experiments on finding the ideological subspace, run the script `src/model/train.sh`.
To replicate the experiments without rotation and sparsity, run the script `src/model/train_ra_sa.sh`.

The scripts expect pickled year-specific datasets in `data/final/`, which can be created using `src/data/prepare_data.sh`.

# Citation

If you use the code in this repository, please cite the following paper:

```
@inproceedings{hofmann2022unsupervised,
    title = {Unsupervised Detection of Contextualized Embedding Bias with Application to Ideology},
    author = {Hofmann, Valentin and Pierrehumbert, Janet and Sch{\"u}tze, Hinrich},
    booktitle = {Proceedings of the 39th International Conference on Machine Learning},
    year = {2022}
}
```