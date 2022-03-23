import tensorflow as tf


def decode_image(image):
    IMAGE_SIZE = [256, 256]
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    return image


def read_tfrecord(example):
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])

    return image


def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord)

    return dataset


def get_gan_dataset(monet_files, photo_files, augment=None, repeat=True, shuffle=True, batch_size=1):
    monet_ds = load_dataset(monet_files)
    photo_ds = load_dataset(photo_files)

    if augment:
        monet_ds = monet_ds.map(augment)
        photo_ds = photo_ds.map(augment)

    if repeat:
        monet_ds = monet_ds.repeat()
        photo_ds = photo_ds.repeat()

    monet_ds = monet_ds.batch(batch_size, drop_remainder=True)
    # Indentifier for every real photo
    ind = tf.data.Dataset.range(7038)
    photo_ds = tf.data.Dataset.zip((photo_ds, ind))
    photo_ds = photo_ds.batch(batch_size, drop_remainder=True)

    monet_ds = monet_ds.prefetch()
    photo_ds = photo_ds.prefetch()

    gan_ds = tf.data.Dataset.zip((monet_ds, photo_ds))

    return gan_ds
