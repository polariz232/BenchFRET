from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate, add
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM, Bidirectional


def conv1d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=True):
    x = Conv1D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


def _os_cnn_block(branch_0, block_filters, pool, max_prime, res=None, activation='relu'):
    if pool:
        branch_0 = conv1d_bn(branch_0, block_filters, 1)

    branches = [conv1d_bn(branch_0, filters=block_filters, kernel_size=1)]
    for number in range(2, max_prime+1):
        if all(number % i != 0 for i in range(2, number)):
            branches.append(conv1d_bn(branch_0, filters=block_filters, kernel_size=number))

    m = Concatenate(axis=-1)(branches)
    m = BatchNormalization()(m)
    if res is not None:
        res = add([res, m])
        x = Activation(activation)(res)
    else:
        x = Activation(activation)(m)
    return x


def _final_cnn(branch_0, block_filters, res=None, activation='relu'):
    branches = [
        conv1d_bn(branch_0, block_filters, 1),
        conv1d_bn(branch_0, block_filters, 2)
    ]
    m = Concatenate(axis=-1)(branches)
    m = BatchNormalization()(m)
    if res is not None:
        res = add([res, m])
        x = Activation(activation)(res)
    else:
        x = Activation(activation)(m)
    return x


def build_model(channels=6, classes=12, model_type="trace_classifier", max_prime=23, resnet=False):
    trace_input = Input(shape=(None, channels))
    n_filters = 32 if model_type == "trace_classifier" else 64
    x = _os_cnn_block(trace_input, n_filters, pool=0, max_prime=max_prime)
    if resnet:  # includes residual connections (only recommended for trace classifiers
        sc = x
        x = _os_cnn_block(x, n_filters, pool=1, max_prime=max_prime, res=None)
        x = _os_cnn_block(x, n_filters, pool=1, max_prime=max_prime, res=sc)
        sc = x
        x = _os_cnn_block(x, n_filters, pool=1, max_prime=max_prime, res=None)
        x = _os_cnn_block(x, n_filters, pool=1, max_prime=max_prime, res=sc)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.5)(x)
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
    elif model_type == "trace_classifier":
        x = _os_cnn_block(x, n_filters, pool=1, max_prime=max_prime)
        x = _final_cnn(x, n_filters)
        x = Dropout(0.1)(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.5)(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
    else:  # state classifier or number of states classifier
        x = _final_cnn(x, 32, res=None)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.5)(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.5)(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation="softmax", kernel_initializer="he_normal")(x)
    model = Model(inputs=trace_input, outputs=x, name="DeepLASI_" + str(channels) + "channel_" + model_type)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model



















