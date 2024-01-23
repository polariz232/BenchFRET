import tensorflow as tf
import numpy as np
from functools import partial
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse


def apply_tsaug(x):
    
    augmenter = (TimeWarp(n_speed_change=5, max_speed_ratio=3) @ 0.5 + Quantize(n_levels=[20, 30, 50, 100]) @ 0.5 + Drift(max_drift=(0.01, 0.1), n_drift_points=5) @ 0.5 + Reverse() @ 0.5)

    x = augmenter.augment(np.array(x))
    
    return x

def tsaug_wrapper(x, y):
    
    # Apply TSAUG augmentations within tf.py_function
    # The input tensors x and y are converted to NumPy arrays and passed to apply_tsaug
    x_augmented = tf.py_function(func=apply_tsaug, inp=[x], Tout=tf.float32)

    # Reshape the output to ensure its shape is correctly set after augmentation
    x_augmented.set_shape(x.get_shape())

    return x_augmented, y

def normalize_tensor(tensor):
    # tensor shape is assumed to be [Batch, Channels, N]

    # Compute the min and max for each channel
    channel_min = tf.reduce_min(tensor, axis=0, keepdims=True)
    channel_max = tf.reduce_max(tensor, axis=0, keepdims=True)

    # Normalize the tensor
    normalized_tensor = (tensor - channel_min) / (channel_max - channel_min)

    # Handling the case where max = min (to avoid division by zero)
    normalized_tensor = tf.where(tf.math.is_nan(normalized_tensor), tf.zeros_like(normalized_tensor), normalized_tensor)

    return normalized_tensor

def preprocess_data(x, y, num_classes=2):
        
    x = normalize_tensor(x)
        
    y = tf.cast(y, tf.int8)
    y = tf.one_hot(y, depth=num_classes)
    
    return x, y


def create_dataset(data, labels, num_classes=2, augment=True, batch_size=32, shuffle_buffer_size=100):
    """
    Creates a TensorFlow dataset from provided data and labels.
    
    Args:
    data: Input data (features).
    labels: Corresponding labels.
    num_classes: Number of classes for one-hot encoding of labels.
    augment: Boolean, whether to apply TSAUG augmentation.
    batch_size: Size of the batches of data.
    shuffle_buffer_size: Buffer size for data shuffling.

    Returns:
    A tf.data.Dataset object.
    """
    
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    
    if augment:
        dataset = dataset.map(lambda x, y: tsaug_wrapper(x,y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
     
    preprocess_func = partial(preprocess_data, num_classes=num_classes)
    
    dataset = dataset.map(lambda x, y: 
                          preprocess_func(x,y),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

