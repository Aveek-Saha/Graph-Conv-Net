# Graph Convolutional Networks
A TensorFlow 2 implementation of Graph Convolutional Networks for classification of nodes from the paper, Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017)

This is my attempt at trying to understand and recreate the neural network from from the paper. You can find the official implementation here: https://github.com/tkipf/gcn

## Requirements
- tensorflow 2
- networkx
- numpy

## Run

To train and test the network with the CORA dataset.

```bash
python train.py
```

## Cite

Please cite this paper if you use this code in your own work:

```
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```
