import tensorflow as tf


def discriminator_loss1(real, generated, mask=None):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(real), real)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.zeros_like(generated), generated)
    if mask is not None:
        generated_loss = generated_loss * mask
    total_disc_loss = real_loss + generated_loss

    return tf.reduce_mean(total_disc_loss * 0.5)


def discriminator_loss2(real, generated, mask=None):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.zeros_like(real), real)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(generated), generated)
    if mask is not None:
        generated_loss = generated_loss * mask
    total_disc_loss = real_loss + generated_loss

    return tf.reduce_mean(total_disc_loss * 0.5)


def generator_loss1(generated, mask=None):
    loss = tf.reduce_mean(
        tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
            tf.ones_like(generated), generated))
    if mask is not None:
        loss = loss * mask

    return tf.reduce_mean(loss)


def generator_loss2(generated, mask=None):
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.zeros_like(generated), generated)
    if mask is not None:
        loss = loss * mask
    return tf.reduce_mean(loss)


def cycle_loss(real_image, cycled_image, LAMBDA):
    loss1 = tf.reduce_sum(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1 * 0.0000152587890625


def identity_loss(real_image, same_image, LAMBDA):
    loss = tf.reduce_sum(tf.abs(real_image - same_image))

    return LAMBDA * 0.5 * loss * 0.0000152587890625
