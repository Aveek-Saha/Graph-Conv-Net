import tensorflow as tf
import numpy as np

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
        print(D_inv)

        self.A_hat = D_inv @ A_hat @ D_inv


        w_init = tf.random_normal_initializer()
        self.W = tf.Variable(initial_value=w_init(
            shape=(input_units, output_units), dtype="float32"), trainable=True)
        

    def call(self, X):

        res = self.A_hat @ X @ self.W

        return tf.nn.relu(res)


A = tf.constant([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]], dtype=tf.float32)

X = np.array([
    [i, -i] for i in range(tf.shape(A)[0])
], dtype=float)

X = tf.cast(X, tf.float32)

gcn = GraphConvolutionLayer(A, 2, 2)

print(gcn(X))
