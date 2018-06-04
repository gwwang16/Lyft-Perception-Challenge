from __future__ import absolute_import, division, print_function

from keras import backend as K
from keras.applications.mobilenet import DepthwiseConv2D, relu6
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          Conv2DTranspose, Input, UpSampling2D, ZeroPadding2D,
                          concatenate)
from keras.models import Model

alpha = 1.0


def MobileNet(input_height, input_width):
    assert input_height // 32 * 32 == input_height
    assert input_width // 32 * 32 == input_width
    depth_multiplier = 1
    img_input = Input(shape=[input_height, input_width, 3], name='image_input')
    # s / 2
    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x_s2 = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    # s / 4
    x = _depthwise_conv_block(
        x_s2, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x_s4 = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    # s /  8
    x = _depthwise_conv_block(
        x_s4, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x_s8 = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    # s / 16
    x = _depthwise_conv_block(
        x_s8, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, block_id=7, rate=12)
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, block_id=8, rate=2)
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, block_id=9, rate=2)
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, block_id=10, rate=2)
    x_s16 = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, block_id=11, rate=2)

    x = _depthwise_conv_block(
        x_s16, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x_s32 = _depthwise_conv_block(
        x, 1024, alpha, depth_multiplier, block_id=13, rate=4)

    return img_input, x_s2, x_s4, x_s8, x_s16, x_s32


def SegMobileNet_u(input_height, input_width, num_classes=21):
    assert input_height // 32 * 32 == input_height
    assert input_width // 32 * 32 == input_width
    img_input, x_s2, x_s4, x_s8, x_s16, x_s32 = MobileNet(
        input_height, input_width)

    x_up16 = concatenate([UpSampling2D()(x_s32), x_s16], axis=-1)
    conv6 = conv_block_simple(x_up16, 256, "conv6_1")

    x_up8 = concatenate([UpSampling2D()(conv6), x_s8], axis=-1)
    conv7 = conv_block_simple(x_up8, 128, "conv7_1")

    x_up4 = concatenate([UpSampling2D()(conv7), x_s4], axis=-1)
    conv8 = conv_block_simple(x_up4, 64, "conv8_1")

    x_up2 = concatenate([UpSampling2D()(conv8), x_s2], axis=-1)
    conv9 = conv_block_simple(x_up2, 32, "conv9_1")

    x = Conv2DTranspose(
        num_classes, (3, 3),
        strides=(2, 2),
        padding='same',
        activation='softmax',
        name='prediction')(conv9)

    return Model(img_input, x)


def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(
        filters, (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        strides=strides,
        name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)

    return Activation(relu6, name=prefix + "_activation")(conv)


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """Adds an initial convolution layer (with batch normalization and relu6).
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = Conv2D(
        filters,
        kernel,
        padding='valid',
        use_bias=False,
        strides=strides,
        name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs,
                          pointwise_conv_filters,
                          alpha,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_id=1,
                          rate=1):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    DONE(see--): Allow dilated depthwise convolutions
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D(
        (3, 3),
        padding='same',
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=False,
        name='conv_dw_%d' % block_id,
        dilation_rate=rate)(inputs)
    x = BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(
        pointwise_conv_filters, (1, 1),
        padding='same',
        use_bias=False,
        strides=(1, 1),
        name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(
        axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)
