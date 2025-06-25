import tensorflow as tf


# create discriminator model
def build_discriminator(inputs, img_size=128):
    num_layers = 4
    if img_size > 128:
        num_layers = 5
    f = [2 ** i for i in range(num_layers)]
    x = inputs
    features = []
    for i in range(0, num_layers):

        if i == 0:
            x = tf.keras.layers.SeparableConv2D(f[i] * img_size, 5, strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)

        else:
            x = tf.keras.layers.SeparableConv2D(f[i] * img_size, 5, strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)

        features.append(x)

    output = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs, outputs=[features, output], name="discriminator")

    return model
