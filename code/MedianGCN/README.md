# MedianGCN
Official PyTorch implementation of MedianGCN and TrimmedGCN in **Understanding Structural Vulnerability in Graph Convolutional Networks** (IJCAI 2021).

[https://www.ijcai.org/proceedings/2021/310](https://www.ijcai.org/proceedings/2021/310)

## Usage
* `models/` contains the implementation of GCN, MedianGCN, TrimmedGCN.
* `utils/` contains the graph processing subroutines (`process.py`) and the aggregation (`reduce.py`).
* `adversarial_edges/` contains the perturbed graph under Nettack attack.

## Example (with DGL or PyG)
The median and trimmed aggregation were initially implemented with pure PyTorch, which are much slower and require higher computation overhead. Here we provide DGL and PyG implementation of Median Convolution (`MedianConv`) for convenience, which also leads to a smaller computation overhead:

+ PyTorch Geometric (PyG)

```python
>>> import torch
>>> from median_pyg import MedianConv

>>> edge_index = torch.as_tensor([[0, 1, 2], [2, 0, 1]])
>>> x = torch.randn(3, 5)
>>> conv = MedianConv(5, 2)
>>> conv(x, edge_index)
tensor([[-0.5138, -1.3301],
        [-0.5138,  0.1693],
        [ 0.2367, -1.3301]], grad_fn=<AddBackward0>)
```

+ Deep Graph Library (DGL)

```python
>>> import dgl
>>> import torch
>>> from median_dgl import MedianConv

>>> g = dgl.graph(([0, 1, 2], [2, 0, 1]))
>>> x = torch.randn(3, 5)
>>> conv = MedianConv(5, 2)
>>> conv(g, x)
tensor([[-0.8558,  0.9913],
        [ 0.1039, -0.1196],
        [ 0.2629,  0.0969]], grad_fn=<AddBackward0>)
```


## Example (with DeepRobust and PyG)
[DeepRobust](https://github.com/DSE-MSU/DeepRobust) is a pytorch adversarial library for attack and defense methods on images and graphs, we provide examples for testing MedianGCN under graph adversarial attacks,
see [test_median_gcn](https://github.com/DSE-MSU/DeepRobust/blob/master/examples/graph/test_median_gcn.py)

## Cite
If you find this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{chen2021understanding,
  title     = {Understanding Structural Vulnerability in Graph Convolutional Networks},
  author    = {Chen, Liang and Li, Jintang and Peng, Qibiao and Liu, Yang and Zheng, Zibin and Yang, Carl},
  booktitle = {IJCAI},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {2249--2255},
  year      = {2021},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2021/310},
  url       = {https://doi.org/10.24963/ijcai.2021/310},
}
```
