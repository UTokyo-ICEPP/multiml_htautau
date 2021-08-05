def setup_tensorflow(seed=None, igpu=0):
    # Tensorflow
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action="ignore", category=DeprecationWarning)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # This is necessary for K.random_normal
    # tf.compat.v1.disable_eager_execution()

    physical_devices = tf.config.list_physical_devices('GPU')
    useGPU = len(physical_devices) > 0
    from multiml import logger
    logger.info(f"useGPU = {useGPU}")

    if isinstance(igpu, int):
        igpu = [igpu]

    # GPU memory option
    if useGPU:
        if tf.__version__[0] == '2':
            gpus = [physical_devices[i] for i in igpu]
            tf.config.set_visible_devices(gpus, 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print('available GPU:', logical_gpus)
        else:
            tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            from keras import backend as K
            sess = tf.Session(config=tf_config)
            K.set_session(sess)

    # Random seed
    import random
    import numpy as np
    if seed is None:
        random.seed(None)
        np.random.seed(None)
        if tf.__version__[0] == '2':
            tf.random.set_seed(None)
        else:
            tf.set_random_seed(None)
    else:
        random.seed(1234 + seed)
        np.random.seed(12345 + seed)
        if tf.__version__[0] == '2':
            tf.random.set_seed(123456 + seed)
        else:
            tf.set_random_seed(123456 + seed)
