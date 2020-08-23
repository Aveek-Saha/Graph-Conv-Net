import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from gcn import *


G = nx.karate_club_graph()


A = nx.convert_matrix.to_numpy_matrix(G)
A = tf.convert_to_tensor(A, tf.float32)
X = tf.eye(*tf.shape(A))

# gcn_1 = GraphConvolutionLayer(A, tf.shape(X)[1], 4)
# out_1 = gcn_1(X)
# gcn_2 = GraphConvolutionLayer(A, tf.shape(out_1)[1], 2)
# out_2 = gcn_2(out_1)


# attr = nx.get_node_attributes(G, 'club')
# colors = ['blue' if x == 'Mr. Hi' else 'green' for x in attr.values()]

# res = tf.transpose(out_2)

# plt.scatter(res[0], res[1], c=colors)
# plt.show()

class GraphConvolution(tf.keras.Model):
    """Combines two GCN layers"""

    def __init__(self, A, num_layers, in_unit, out_units, name="graph_convolution",
                 **kwargs
                 ):
        super(GraphConvolution, self).__init__(name=name)

        self.num_layers = num_layers

        gcn = []
        for i in range(num_layers):
            if i == 0:
                gcl = GraphConvolutionLayer(A, in_unit, out_units[i])
            else:
                gcl = GraphConvolutionLayer(A, out_units[i-1], out_units[i])
            gcn.append(gcl)

        self.gcn = gcn
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, X):

        output = X

        for i in range(self.num_layers):
            output = self.gcn[i](output)
        
        output = self.dense(output)

        return output


graph_conv = GraphConvolution(A, 2, tf.shape(X)[1], [4, 2])
out = graph_conv(X)
print(out)
