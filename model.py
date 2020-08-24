import tensorflow as tf
import numpy as np
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from gcn import *

class GraphConvolution(tf.keras.Model):
    """Combines GCN layers"""

    def __init__(self, A, out_units, activations, rate=0.0, l2=0.0, name="graph_convolution"):
        super(GraphConvolution, self).__init__(name=name)

        self.num_layers = len(out_units)
        gcn = []
        for i in range(len(out_units)):
            gcl = GraphConvolutionLayer(out_units[i], A, activations[i], rate, l2)
            gcn.append(gcl)

        self.gcn = gcn

    def call(self, X):

        output = X

        for i in range(self.num_layers):
            output = self.gcn[i](output)

        return output
