import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def norm_adjacency_matrix(A):
  I = tf.eye(tf.shape(A)[0])
  A_hat = A + I

  D_inv = tf.linalg.tensor_diag(
      tf.pow(tf.reduce_sum(A_hat, 0), tf.cast(-0.5, tf.float32)))

  A_hat = D_inv @ A_hat @ D_inv

  return A_hat


class GraphConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, units, A, activation=tf.identity, rate=0.0, l2=0.0):
        super(GraphConvolutionLayer, self).__init__()

        self.activation = activation
        self.units = units
        self.rate = rate
        self.l2 = l2
        self.A = A

    def build(self, input_shape):
        self.W = self.add_weight(
          shape=(input_shape[1], self.units),
          dtype=self.dtype,
          initializer='glorot_uniform',
          regularizer=tf.keras.regularizers.l2(self.l2)
        )

    def call(self, X):

        X = tf.nn.dropout(X, self.rate)
        X = self.A @ X @ self.W
        return self.activation(X)
        
