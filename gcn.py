import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# A = tf.constant([
#     [0, 1, 0, 0],
#     [0, 0, 1, 1],
#     [0, 1, 0, 0],
#     [1, 0, 1, 0]], dtype=tf.float32)

# X = np.array([
#     [i, -i] for i in range(tf.shape(A)[0])
# ], dtype=float)

# X = tf.cast(X, tf.float32)

# I = tf.eye(tf.shape(A)[0])

# A_hat = A + I

# D = tf.linalg.tensor_diag(tf.reduce_sum(A, 0))
# D_inverse = tf.linalg.inv(D)

# w_init = tf.random_normal_initializer()

# W = w_init(shape=(2, 2), dtype=tf.float32)

# res = D_inverse @ A @ X @ W
# print(res)


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

        return tf.nn.relu(res)


# A = tf.constant([
#     [0, 1, 0, 0],
#     [0, 0, 1, 1],
#     [0, 1, 0, 0],
#     [1, 0, 1, 0]], dtype=tf.float32)

# X = np.array([
#     [i, -i] for i in range(tf.shape(A)[0])
# ], dtype=float)

# X = tf.cast(X, tf.float32)

# gcn = GraphConvolutionLayer(A, 2, 2)

# print(gcn(X))

G = nx.karate_club_graph()

# print(nx.adjacency_matrix(G).todense())
# print(nx.get_node_attributes(G, 'club'))

A = nx.convert_matrix.to_numpy_matrix(G)
A = tf.convert_to_tensor(A, tf.float32)
X = tf.eye(*tf.shape(A))

gcn_1 = GraphConvolutionLayer(A, tf.shape(X)[1], 4)
out_1 = gcn_1(X)
gcn_2 = GraphConvolutionLayer(A, tf.shape(out_1)[1], 2)
out_2 = gcn_2(out_1)

# print(tf.transpose(out_2))
# print(out_2)

attr = nx.get_node_attributes(G, 'club')
colors = ['blue' if x == 'Mr. Hi' else 'green' for x in attr.values()]
# print(colors)

res = tf.transpose(out_2)

plt.scatter(res[0], res[1], c=colors)
plt.show()
