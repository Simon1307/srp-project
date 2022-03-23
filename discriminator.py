import tensorflow as tf
from model_utils import down_sample
import tensorflow_addons as tfa


def Monet_Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    x = inp

    down1 = down_sample(64, 4, False)(x)  # (size, 128, 128, 64)
    down2 = down_sample(128, 4)(down1)  # (size, 64, 64, 128)
    down3 = down_sample(256, 4)(down2)  # (size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False) \
        (zero_pad1)  # (size, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (size, 33, 33, 512)

    return tf.keras.Model(inputs=inp, outputs=zero_pad2)


def Monet_Output_Layer():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[33, 33, 512], name='input_image')
    x = inp

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)  # (size, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)


def Photo_Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    x = inp

    down1 = down_sample(64, 4, False)(x)  # (size, 128, 128, 64)
    down2 = down_sample(128, 4)(down1)  # (size, 64, 64, 128)
    down3 = down_sample(256, 4)(down2)  # (size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(
        zero_pad1)  # (size, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (size, 33, 33, 512)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (size, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)


def Discriminator_Guess():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 6], name='input_image')
    x = inp

    down1 = down_sample(64, 4, False)(x)  # (size, 128, 128, 64)
    down2 = down_sample(128, 4)(down1)  # (size, 64, 64, 128)
    down3 = down_sample(256, 4)(down2)  # (size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(
        zero_pad1)  # (size, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (size, 33, 33, 512)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (size, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)
