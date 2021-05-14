def noise_layer(x):
    from tensorflow.keras import backend as K
    x = x * 0 + K.random_normal(shape=K.shape(x), mean=0.0, stddev=1.0)
    return x


def zero_layer(x):
    x = x * 0
    return x
