import tensorflow as tf
import numpy as np

A = tf.constant([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]], dtype=tf.float32)

X = np.array([
    [i, -i] for i in range(tf.shape(A)[0])
], dtype=float)

X = tf.cast(X, tf.float32)

I = tf.eye(tf.shape(A)[0])

A_hat = A + I

D = tf.linalg.tensor_diag(tf.reduce_sum(A, 0))
D_inverse = tf.linalg.inv(D)

w_init = tf.random_normal_initializer()

W = w_init(shape=(2, 2), dtype=tf.float32)

res = D_inverse @ A @ X @ W
print(res)
