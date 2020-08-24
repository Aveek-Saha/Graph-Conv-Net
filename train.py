import tensorflow as tf
import numpy as np
import os
import csv

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

data = []
edges = []

with open(os.path.join("cora","cora.content")) as tsv:
    for line in csv.reader(tsv, delimiter="\t"):
      data.append(line)

with open(os.path.join("cora","cora.cites")) as tsv:
    for line in csv.reader(tsv, delimiter="\t"):
      edges.append(line)

data = shuffle(data,random_state=1)


labels = []
nodes = []
features = []

for row in data:
  labels.append(row[-1])
  features.append(row[1:-1])
  nodes.append(row[0])

features = np.array(features,dtype=int)

edge_list=[]
for edge in edges:
    edge_list.append((edge[0],edge[1]))


num_nodes = features.shape[0]

test = int(0.35 * num_nodes)
val = int(0.18 * num_nodes)

index = [i for i in range(num_nodes)]
index = shuffle(index, random_state=1)

train_index = index[:(num_nodes-test-val)]
val_index = index[(num_nodes-test-val):(num_nodes-test)]
test_index = index[(num_nodes-test):]

# len(train_index), len(val_index), len(test_index)

train_mask = np.zeros((num_nodes,), dtype=bool)
train_mask[train_index] = True

val_mask = np.zeros((num_nodes,), dtype=bool)
val_mask[val_index] = True

test_mask = np.zeros((num_nodes,), dtype=bool)
test_mask[test_index] = True


def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels)
    return labels, label_encoder.classes_


labels_encoded, classes = encode_label(labels)
