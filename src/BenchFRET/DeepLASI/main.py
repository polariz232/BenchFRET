import pickle
import os
import numpy as np
import importlib
import importlib.resources as resources
from DeepGapSeq.DeepLASI.dataloader import create_dataset
from DeepGapSeq.DeepLASI.deepLASI_architectures import build_model
import tensorflow as tf

module_path = resources.files(importlib.import_module(f'DeepGapSeq.InceptionFRET'))
dataset_path = os.path.join(module_path, "deepgapseq_simulated_traces","dataset_2023_11_22.pkl")

with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)
    
data = dataset["data"]
labels = dataset["labels"]

data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

num_classes = 2    
n_channels = 2

batch_size = 32
learning_rate = 0.001
epochs = 10


dataset = create_dataset(data, 
                         labels, 
                         num_classes = num_classes, 
                         augment=False,
                         batch_size=batch_size)

# for features, labels in dataset.take(1):
#     print("Features shape:", features.shape)
#     print("Labels shape:", labels.shape)


model = build_model(channels=n_channels, classes=num_classes)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(dataset, epochs=epochs)
