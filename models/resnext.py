import tensorflow as tf
from keras.utils.layer_utils import get_source_inputs
from keras import backend

import collections

ModelParams = collections.namedtuple(
    'ModelParams',
    ['model_name', 'repetitions']
)


def batchnorm_relu(x):
    """ Batch Normalization & ReLU """
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def residual_block(inputs, num_filters, kernel_size=4, strides=1):
    """ Convolutional Layers """
    x = batchnorm_relu(inputs)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same", strides=1)(x)

    """ Shortcut Connection (Identity Mapping) """
    s = tf.keras.layers.Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)

    """ Addition """
    x = tf.keras.layers.Add()([x, s])
    return x


def conv_block_2nd(input, num_filters, ks=4):
    x = tf.keras.layers.Conv2D(num_filters, ks, padding='same')(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(num_filters, ks, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    return x


def decoder_block(x, skip_features, num_filters, feature_extract=False, kernel_size=4):
    if feature_extract:
        skip_features = tf.keras.layers.Conv2D(num_filters, 4, padding="same", name="feature_extractor_extra")(
            skip_features)
        skip_features = tf.keras.layers.LeakyReLU()(skip_features)

    x = tf.keras.layers.Conv2DTranspose(num_filters, (4, 4), strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, skip_features])

    x = residual_block(x, num_filters, kernel_size)

    return x


def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'glorot_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    axis = 3 if backend.image_data_format() == 'channels_last' else 1
    default_bn_params = {
        'axis': axis,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params


def slice_tensor(x, start, stop, axis):
    if axis == 3:
        return x[:, :, :, start:stop]
    elif axis == 1:
        return x[:, start:stop, :, :]
    else:
        raise ValueError("Slice axis should be in (1, 3), got {}.".format(axis))


# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------

def GroupConv2D(filters,
                kernel_size,
                strides=(1, 1),
                groups=32,
                kernel_initializer='he_uniform',
                use_bias=True,
                activation='linear',
                padding='valid',
                **kwargs):
    """
    Grouped Convolution Layer implemented as a Slice,
    Conv2D and Concatenate layers. Split filters to groups, apply Conv2D and concatenate back.
    Args:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the stride
            length of the convolution.
        groups: Integer, number of groups to split input filters to.
        kernel_initializer: Regularizer function applied to the kernel weights matrix.
        use_bias: Boolean, whether the layer uses a bias vector.
        activation: Activation function to use (see activations).
            If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        padding: one of "valid" or "same" (case-insensitive).
    Input shape:
        4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".
    Output shape:
        4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format is "channels_last".
        rows and cols values might have changed due to padding.
    """

    slice_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        inp_ch = int(backend.int_shape(input_tensor)[-1] // groups)  # input grouped channels
        out_ch = int(filters // groups)  # output grouped channels

        blocks = []
        for c in range(groups):
            slice_arguments = {
                'start': c * inp_ch,
                'stop': (c + 1) * inp_ch,
                'axis': slice_axis,
            }
            x = tf.keras.layers.Lambda(slice_tensor, arguments=slice_arguments)(input_tensor)
            x = tf.keras.layers.Conv2D(out_ch,
                                       kernel_size,
                                       strides=strides,
                                       kernel_initializer=kernel_initializer,
                                       use_bias=use_bias,
                                       activation=activation,
                                       padding=padding)(x)
            blocks.append(x)

        x = tf.keras.layers.Concatenate(axis=slice_axis)(blocks)
        return x

    return layer


def conv_block(filters, stage, block, strides=(2, 2), **kwargs):
    """The conv block is the block that has conv layer at shortcut.
    # Arguments
        filters: integer, used for first and second conv layers, third conv layer double this value
        strides: tuple of integers, strides for conv (3x3) layer in block
        stage: integer, current stage label, used for generating layer names
        block: integer, current block label, used for generating layer names
    # Returns
        Output layer for the block.
    """

    def layer(input_tensor):
        # extracting params and names for layers
        conv_params = get_conv_params()
        group_conv_params = dict(list(conv_params.items()) + list(kwargs.items()))
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = tf.keras.layers.Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(input_tensor)
        x = tf.keras.layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = tf.keras.layers.LeakyReLU(name=relu_name + '1')(x)

        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), strides=strides, **group_conv_params)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = tf.keras.layers.LeakyReLU(name=relu_name + '2')(x)

        x = tf.keras.layers.Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name + '3', **bn_params)(x)

        shortcut = tf.keras.layers.Conv2D(filters * 2, (1, 1), name=sc_name, strides=strides, **conv_params)(
            input_tensor)
        shortcut = tf.keras.layers.BatchNormalization(name=sc_name + '_bn', **bn_params)(shortcut)
        x = tf.keras.layers.Add()([x, shortcut])

        x = tf.keras.layers.LeakyReLU(name=relu_name)(x)

        return x

    return layer


def identity_block(filters, stage, block, **kwargs):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        filters: integer, used for first and second conv layers, third conv layer double this value
        stage: integer, current stage label, used for generating layer names
        block: integer, current block label, used for generating layer names
    # Returns
        Output layer for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        group_conv_params = dict(list(conv_params.items()) + list(kwargs.items()))
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = tf.keras.layers.Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(input_tensor)
        x = tf.keras.layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = tf.keras.layers.LeakyReLU(name=relu_name + '1')(x)

        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), **group_conv_params)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = tf.keras.layers.LeakyReLU(name=relu_name + '2')(x)

        x = tf.keras.layers.Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name + '3', **bn_params)(x)

        x = tf.keras.layers.Add()([x, input_tensor])

        x = tf.keras.layers.LeakyReLU(name=relu_name)(x)

        return x

    return layer


def ResNeXt(
        model_params,
        include_top=False,
        input_tensor=None,
        input_shape=None,
        classes=1000,
        weights='imagenet',
        **kwargs):
    """Instantiates the ResNet, SEResNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Args:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns:
        A Keras model instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape, name='data')
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()

    # resnext bottom
    x = tf.keras.layers.BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(x)
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = tf.keras.layers.BatchNormalization(name='bn0', **bn_params)(x)
    # x = tf.keras.layers.Activation('relu', name='relu0')(x)
    x = tf.keras.layers.LeakyReLU(name='relu0')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # resnext body
    init_filters = 128
    for stage, rep in enumerate(model_params.repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if stage == 0 and block == 0:
                x = conv_block(filters, stage, block, strides=(1, 1), **kwargs)(x)

            elif block == 0:
                x = conv_block(filters, stage, block, strides=(2, 2), **kwargs)(x)

            else:
                x = identity_block(filters, stage, block, **kwargs)(x)

    # resnext top
    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D(name='pool1')(x)
        x = tf.keras.layers.Dense(classes, name='fc1')(x)
        x = tf.keras.layers.Activation('softmax', name='softmax')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = tf.keras.models.Model(inputs, x)

    return model


# -------------------------------------------------------------------------
#   Residual Models
# -------------------------------------------------------------------------

MODELS_PARAMS = {
    'resnext50': ModelParams('resnext50', (3, 4, 6, 3)),
    'resnext101': ModelParams('resnext101', (3, 4, 23, 3)),
}


def ResNeXt50(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNeXt(
        MODELS_PARAMS['resnext50'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def ResNeXt101(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNeXt(
        MODELS_PARAMS['resnext101'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def build_resnext50_unet(input_shape, img_size=128, img_channel=3, kersize=4):
    inputs = tf.keras.layers.Input(input_shape, name="input_1")

    resnext50 = ResNeXt50(include_top=False, weights=None, input_tensor=inputs)

    """ Encoder """
    s1 = resnext50.get_layer("input_1").output  ## (256 x 256)
    s2 = resnext50.get_layer("relu0").output  ## (128 x 128)
    s3 = resnext50.get_layer("stage2_unit1_relu1").output  ## (64 x 64)
    s4 = resnext50.get_layer("stage3_unit1_relu1").output  ## (32 x 32)
    s5 = resnext50.get_layer("stage4_unit1_relu1").output  ## (16 x 16)

    """ Bridge """
    b1 = resnext50.get_layer("stage4_unit3_relu").output  ## (8 x 8)

    """ Decoder """
    x = img_size  # (16 x 16)
    d1 = decoder_block(b1, s5, x, kernel_size=kersize)
    x = x / 2  # (32 x 32)
    d2 = decoder_block(d1, s4, x, kernel_size=kersize)
    x = x / 2  # (64 x 64)
    d3 = decoder_block(d2, s3, x, kernel_size=kersize)
    x = x / 2  # (128 x 128)
    d4 = decoder_block(d3, s2, x, kernel_size=kersize)
    x = x / 2  # (256 x 256)
    d5 = decoder_block(d4, s1, x, feature_extract=True)

    """ Output """
    outputs = tf.keras.layers.Conv2D(img_channel, 1, padding="same", activation="tanh")(d5)

    model = tf.keras.models.Model(inputs, outputs, name="ResNext50_U-Net")

    return model
