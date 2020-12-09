import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
    UpSampling2D, Cropping2D, concatenate, ZeroPadding2D, SpatialDropout2D

import functools

def create(input_shape, num_class=1, activation=tf.nn.relu):
    opts = locals().copy()

    # model = Depth_BBBP(num_class, activation)
    # return model, opts

    concat_axis = 3
    inputs = tf.keras.layers.Input(shape=input_shape)

    Conv2D_ = functools.partial(tfp.layers.Convolution2DReparameterization, activation=activation, padding='same')

    conv1 = Conv2D_(32, (3, 3))(inputs)
    conv1 = Conv2D_(32, (3, 3))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D_(64, (3, 3))(pool1)
    conv2 = Conv2D_(64, (3, 3))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D_(128, (3, 3))(pool2)
    conv3 = Conv2D_(128, (3, 3))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D_(256, (3, 3))(pool3)
    conv4 = Conv2D_(256, (3, 3))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D_(512, (3, 3))(pool4)
    conv5 = Conv2D_(512, (3, 3))(conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = Conv2D_(256, (3, 3))(up6)
    conv6 = Conv2D_(256, (3, 3))(conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv2D_(128, (3, 3))(up7)
    conv7 = Conv2D_(128, (3, 3))(conv7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D_(64, (3, 3))(up8)
    conv8 = Conv2D_(64, (3, 3))(conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D_(32, (3, 3))(up9)
    conv9 = Conv2D_(32, (3, 3))(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    conv10 = Conv2D(num_class, (1, 1))(conv9)
    conv10 = 1e-6 * conv10

    model = tf.keras.models.Model(inputs=inputs, outputs=conv10)
    return model, opts

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)
#
# # import numpy as np
# # model = create((64,64,3), 2)
# # x = np.ones((1,64,64,3), dtype=np.float32)
# # output = model(x)
# # import pdb; pdb.set_trace()
