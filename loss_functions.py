import tensorflow as tf


def discriminator_loss1(real, generated):
    real_loss = tf.math.minimum(tf.zeros_like(real), real - tf.ones_like(real))

    generated_loss = tf.math.minimum(tf.zeros_like(generated), - generated - tf.ones_like(generated))

    total_disc_loss = real_loss + generated_loss

    return tf.reduce_mean(-total_disc_loss * 0.5)


def discriminator_loss2(real, generated):
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) \
        (tf.ones_like(generated), generated)
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) \
        (tf.zeros_like(real), real)
    total_disc_loss = real_loss + generated_loss

    return tf.reduce_mean(total_disc_loss * 0.5)


def generator_loss1(generated):
    return tf.reduce_mean(-generated)


def generator_loss2(generated):
    return tf.reduce_mean(
        tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
            tf.zeros_like(generated), generated))


def cycle_loss(real_image, cycled_image, LAMBDA):
    loss1 = tf.reduce_sum(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1 * 0.0000152587890625


def identity_loss(real_image, same_image, LAMBDA):
    loss = tf.reduce_sum(tf.abs(real_image - same_image))

    return LAMBDA * 0.5 * loss * 0.0000152587890625

