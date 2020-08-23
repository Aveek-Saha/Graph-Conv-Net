import tensorflow as tf
import numpy as np
import networkx as nx
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from gcn import *


G = nx.karate_club_graph()

A = nx.convert_matrix.to_numpy_matrix(G)
A = tf.convert_to_tensor(A, tf.float32)
# X = tf.eye(*tf.shape(A))

X = np.zeros((A.shape[0], 2))
node_distance_instructor = nx.shortest_path_length(G, target=33)
node_distance_administrator = nx.shortest_path_length(G, target=0)

for node in G.nodes():
    X[node][0] = node_distance_administrator[node]
    X[node][1] = node_distance_instructor[node]

attr = nx.get_node_attributes(G, 'club')
y = np.array([0 if x == 'Mr. Hi' else 1 for x in attr.values()])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

feat_dim = X.shape

X_train = tf.convert_to_tensor(X_train, tf.float32)
X_test = tf.convert_to_tensor(X_test, tf.float32)
y_train = tf.convert_to_tensor(y_train, tf.float32)
y_test = tf.convert_to_tensor(y_test, tf.float32)
# print(X_train)


class GraphConvolution(tf.keras.Model):
    """Combines GCN layers"""

    def __init__(self, A, out_units, rate=0.0,
                 l2=0.0, name="graph_convolution"):
        super(GraphConvolution, self).__init__(name=name)

        gcn = []
        for i in range(len(out_units)):
            gcl = GraphConvolutionLayer(A, out_units[i], rate, l2)
            gcn.append(gcl)

        self.gcn = gcn
        self.num_layers = len(out_units)
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, X):

        output = X

        for i in range(self.num_layers):
            output = self.gcn[i](output)

        output = self.dense(output)

        return output


# inp = tf.keras.Input((feat_dim[1]))
# out_1 = GraphConvolutionLayer(A, feat_dim[1], 2)(inp)
# out_2 = GraphConvolutionLayer(A, 2, 1)(out_1)
# out = tf.keras.layers.Dense(1, activation="sigmoid")(out_2)

# model = tf.keras.Model(inputs=inp, outputs=out, name="graph_convolution")

# optimizer = tf.keras.optimizers.SGD()

# model.compile(optimizer, tf.keras.losses.BinaryCrossentropy())

# model.fit(X, y, epochs=100, batch_size=34)
A = norm_adjacency_matrix(A)

l2 = 5e-3
rate = 0.5
epochs = 1000

gcn = GraphConvolution(A, [4, 2], rate, l2)

bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
loss_metric = tf.keras.metrics.Mean()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Iterate over epochs.

for epoch in range(epochs):
    # print("Start of epoch %d" % (epoch,))

    with tf.GradientTape() as tape:
        reconstructed = gcn(X)
        # Compute reconstruction loss
        loss = bce_loss_fn(y, reconstructed)

    grads = tape.gradient(loss, gcn.trainable_weights)
    optimizer.apply_gradients(zip(grads, gcn.trainable_weights))

    loss_metric(loss)

    if epoch % 100 == 0:
        print("epoch %d: mean loss = %.4f" % (epoch, loss_metric.result()))

res = gcn.predict(X, batch_size=34)
y_pred = res.flatten()
y_pred = np.where(y_pred >= 0.5, 1, 0)
print(classification_report(y, y_pred))