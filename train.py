import tensorflow as tf
import numpy as np
import os
import csv

from gcn import *

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
num_labels = len(labels)


def get_index(num_labels, num_train_class=20, num_test=1000, num_val=500):
  
  label_count = {}
  train_index = []
  for i, label in enumerate(labels):
    if not label in label_count:
      label_count[label] = 1
      train_index.append(i)
    else:
      if label_count[label] >= num_train_class:
        continue
      else:
        label_count[label] += 1
        train_index.append(i)

  test_index = []
  count=0
  for i in range(num_labels):
    if count >= num_test:
      break
    if i not in train_index:
      test_index.append(i)
      count += 1

  val_index = []
  count=0
  for i in range(num_labels):
    if count >= num_val:
      break
    if i not in train_index and i not in test_index:
      val_index.append(i)
      count += 1

  return train_index, test_index, val_index

num_nodes = features.shape[0]

test = 1000
val = 500
train = 20

# train_index, test_index, val_index = get_index(num_labels, train, test, val)

index = [i for i in range(num_nodes)]
index = shuffle(index,random_state=1)

train_index = index[:(num_nodes-test-val)] 
val_index = index[(num_nodes-test-val):(num_nodes-test)] 
test_index = index[(num_nodes-test):]

len(train_index), len(val_index), len(test_index)

train_mask = np.zeros((num_nodes,),dtype=bool)
train_mask[train_index] = True

val_mask = np.zeros((num_nodes,),dtype=bool)
val_mask[val_index] = True

test_mask = np.zeros((num_nodes,),dtype=bool)
test_mask[test_index] = True

def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels)
    return labels, label_encoder.classes_


G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edge_list)

A = nx.convert_matrix.to_numpy_matrix(G)
A = tf.convert_to_tensor(A, tf.float32)
print('Graph info: ', nx.info(G))

A = norm_adjacency_matrix(A)

l2 = 5e-4
rate = 0.5
epochs = 200
learning_rate = 1e-2 
labels_encoded, classes = encode_label(labels)


inp = tf.keras.Input((features.shape[1],))
out_1 = GraphConvolutionLayer(16, A, rate=rate, l2=l2)(inp)
out = GraphConvolutionLayer( 7, A, tf.nn.sigmoid, rate)(out_1)

model = tf.keras.Model(inputs= inp, outputs=out, name="graph_convolution")

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])

print(model.summary())

validation_data = (features, labels_encoded, val_mask)
model.fit(features,
          labels_encoded,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=num_nodes,
          validation_data=validation_data,
          shuffle=False)


features_test = features[test_mask]
A_test = np.array(A)[test_mask,:][:,test_mask]
y_test = labels_encoded[test_mask]

y_pred = model.predict(features, batch_size=num_nodes)

report = classification_report(np.argmax(y_test,axis=1), np.argmax(y_pred[test_mask],axis=1), target_names=classes)
print('GCN Classification Report: \n {}'.format(report))
