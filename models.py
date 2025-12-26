import tensorflow as tf


def build_xception(input_shape=(128, 128, 3), pretrained='imagenet', freeze_base=False):
    """Build a binary classifier using Xception backbone.

    Returns a compiled `tf.keras.Model` ready for training.
    """
    base = tf.keras.applications.Xception(
        include_top=False, weights=pretrained, input_shape=input_shape, pooling='avg'
    )
    if freeze_base:
        base.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base(x, training=not freeze_base)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model
