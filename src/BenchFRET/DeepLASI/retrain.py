import pickle
import os
import numpy as np
import importlib
import importlib.resources as resources
from BenchFRET.DeepLASI.dataloader import create_dataset
from BenchFRET.DeepLASI.deepLASI_architectures import build_model
from BenchFRET.pipeline.dataloader import SimLoader
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

n_states = 2    
n_colors = 2
ratio_train = 0.7
val_test_split = 0.5
batch_size = 32
learning_rate = 0.001
epochs = 10

dataset_path = r'F:\retrain_dataset\pickledict\data.pkl'
output_directory = './retrained_models/'
    
simloader = SimLoader(data_path=dataset_path)
data, _ = simloader.get_data()
labels = simloader.get_labels()

X = np.array(data, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      train_size=ratio_train,
                                                      random_state=42,
                                                      shuffle=True)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,
                                                train_size=val_test_split,
                                                random_state=42,
                                                shuffle=True)

training_dataset = create_dataset(X_train,
                                y_train,
                                num_classes = n_states, 
                                augment=False,
                                batch_size=batch_size)

validation_dataset = create_dataset(X_val,
                                y_val,
                                num_classes = n_states, 
                                augment=False,
                                batch_size=batch_size)

test_dataset = create_dataset(X_test,
                                y_test,
                                num_classes = n_states, 
                                augment=False,
                                batch_size=batch_size)



model_directory = resources.files(importlib.import_module(f'BenchFRET.DeepLASI'))

state_model_name = "DeepLASI_{}color_{}state_classifier.h5".format(n_colors, n_states)
state_model_path = os.path.join(model_directory, "models", state_model_name)
model = tf.keras.models.load_model(state_model_path)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=output_directory, monitor='loss',
                                                    save_best_only=True)
earlystopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

callbacks = [reduce_lr, model_checkpoint, earlystopping]

hist = model.fit(training_dataset, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    verbose=True, 
                    validation_data=validation_dataset,
                    callbacks=callbacks)

model.save(os.path.join(output_directory, 'model.h5'))
 