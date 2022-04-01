
# Neo-GNN - Official PyTorch Implementation

This repository provides the official PyTorch implementation for the following paper:

**Neo-GNNs: Neighborhood Overlap-aware Graph Neural Networks for Link Prediction**<br>
Seongjun Yun, Seoyoon Kim, Junhyun Lee, Jaewoo Kang, Hyunwoo J. Kim<br>
In NeurIPS 2021.<br>
[**Paper**](https://openreview.net/forum?id=Ic9vRN3VpZ) 
> **Abstract:** *Graph Neural Networks (GNNs) have been widely applied to various fields for learning over graph-structured data. They have shown significant improvements over traditional heuristic methods in various tasks such as node classification and graph classification. However, since GNNs heavily rely on smoothed node features rather than graph structure, they often show poor performance than simple heuristic methods in link prediction where the structural information, e.g., overlapped neighborhoods, degrees, and shortest paths, is crucial. To address this limitation, we propose Neighborhood Overlap-aware Graph Neural Networks (Neo-GNNs) that learn useful structural features from an adjacency matrix and estimate overlapped neighborhoods for link prediction. Our Neo-GNNs generalize neighborhood overlap-based heuristic methods and handle overlapped multi-hop neighborhoods. Our extensive experiments on Open Graph Benchmark datasets (OGB) demonstrate that Neo-GNNs consistently achieve state-of-the-art performance in link prediction.*


Requirements
------------

Latest combination: Python 3.8.8 + PyTorch 1.8.1 + PyTorch\_Geometric 2.0.3 + OGB 1.3.2.

Install [PyTorch](https://pytorch.org/)

Install [PyTorch\_Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Install [OGB](https://ogb.stanford.edu/docs/home/)

We used the code of [pytorch_indexing](https://github.com/taylorpatti/pytorch_indexing) embedded in our project for sparse matrix computation that can be backpropagated.

GPU with more than 24GB memory. (We used GeForce RTX 3090 & Qudadro RTX 8000)

Usages
------

### ogbl-collab

    python main_collab.py

### ogbl-ppa

    python main_ppa.py

### ogbl-ddi

    python main_ddi.py

Reference
---------

If you find the code useful, please cite our papers.

    @inproceedings{yun2021neognns,
		title={Neo-{GNN}s: Neighborhood Overlap-aware Graph Neural Networks for Link Prediction},
		author={Seongjun Yun and Seoyoon Kim and Junhyun Lee and Jaewoo Kang and Hyunwoo J. Kim},
		booktitle={Advances in Neural Information Processing Systems},
		editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
		year={2021},
		url={https://openreview.net/forum?id=Ic9vRN3VpZ}
	}
