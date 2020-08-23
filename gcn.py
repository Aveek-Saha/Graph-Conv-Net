import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def norm_adjacency_matrix(A):
  I = tf.eye(tf.shape(A)[0])
  A_hat = A + I

  D_inv = tf.linalg.tensor_diag(
      tf.pow(tf.reduce_sum(A, 0), tf.cast(-0.5, tf.float32)))

  A_hat = D_inv @ A_hat @ D_inv

  return A_hat

class GraphConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, A, input_units, output_units):
        super(GraphConvolutionLayer, self).__init__()

        I = tf.eye(tf.shape(A)[0])
        A_hat = A + I

        D_inv = tf.linalg.tensor_diag(
            tf.pow(tf.reduce_sum(A, 0), tf.cast(-0.5, tf.float32)))

        self.A_hat = D_inv @ A_hat @ D_inv

        w_init = tf.random_uniform_initializer(-1, 1)
        self.W = tf.Variable(initial_value=w_init(
            shape=(input_units, output_units), dtype="float32"), trainable=True)

    def call(self, X):
        res = self.A_hat @ X @ self.W
        return tf.nn.tanh(res)

