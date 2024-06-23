import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE

import time


def flip_left_right(lowres_img, highres_img):
    """Flips Images to left and right."""

    # Outputs random values from a uniform distribution in between 0 to 1
    rn = tf.random.uniform(shape=(), maxval=1)
    # If rn is less than 0.5 it returns original lowres_img and highres_img
    # If rn is greater than 0.5 it returns flipped image
    return tf.cond(
        rn < 0.5,
        lambda: (lowres_img, highres_img),
        lambda: (
            tf.image.flip_left_right(lowres_img),
            tf.image.flip_left_right(highres_img),
        ),
    )


def noise(lowres_img, highres_img):
    """Adds random noise to each image in the supplied array."""
    lowres_img = tf.cast(lowres_img, tf.int16)
    original_size = tf.shape(lowres_img)

    n = tf.cast(tf.random.normal(original_size, mean=0.0, stddev=10), tf.int16)
    noisy_image = lowres_img + n

    return tf.cast(tf.clip_by_value(noisy_image, 0, 255), tf.uint8), highres_img


def random_rotate(lowres_img, highres_img):
    """Rotates Images by 90 degrees."""

    # Outputs random values from uniform distribution in between 0 to 4
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    # Here rn signifies number of times the image(s) are rotated by 90 degrees
    return tf.image.rot90(lowres_img, rn), tf.image.rot90(highres_img, rn)


def random_rescale(image_hr, hr_crop_size, scale):
    """
    Rescales the image to a different size.
    Then rescales it back to the original size.
    Losing some information.
    """

    image_lr = tf.image.resize(
        image_hr,
        (hr_crop_size // scale, hr_crop_size // scale),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    image_lr = tf.image.resize(
        image_lr,
        (hr_crop_size, hr_crop_size),
        method=tf.image.ResizeMethod.BILINEAR,
    )

    return image_lr


def random_crop(_, highres_img, hr_crop_size=256, scale=2):
    """Crop images.

    low resolution images: 64x64
    high resolution images: 256x256
    """

    # lowres_img, highres_img = random_rescale(highres_img, _lowres_img_scale)
    # lowres_crop_size = hr_crop_size // scale  # 96//4=24
    # lowres_img_shape = tf.shape(lowres_img)[:2]  # (height,width)

    # lowres_width = tf.random.uniform(
    #     shape=(), maxval=lowres_img_shape[1] - lowres_crop_size + 1, dtype=tf.int32
    # )
    # lowres_height = tf.random.uniform(
    #     shape=(), maxval=lowres_img_shape[0] - lowres_crop_size + 1, dtype=tf.int32
    # )

    # highres_width = lowres_width * scale
    # highres_height = lowres_height * scale

    hr_shape = tf.shape(highres_img)[:2]
    width = tf.random.uniform(
        shape=(), maxval=hr_shape[1] - hr_crop_size + 1, dtype=tf.int32
    )
    height = tf.random.uniform(
        shape=(), maxval=hr_shape[0] - hr_crop_size + 1, dtype=tf.int32
    )
    highres_img_cropped = highres_img[
        height : height + hr_crop_size, width : width + hr_crop_size
    ]
    lowres_img = random_rescale(highres_img_cropped, hr_crop_size, scale)

    return lowres_img, highres_img_cropped


def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)
    return psnr_value


def dataset_object(dataset_cache, training=True):
    ds = dataset_cache
    ds = ds.map(
        lambda lowres, highres: random_crop(lowres, highres, scale=4),
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
        ds = ds.map(flip_left_right, num_parallel_calls=AUTOTUNE)
        ds = ds.map(noise, num_parallel_calls=AUTOTUNE)
    # Batching Data
    ds = ds.batch(16)

    if training:
        # Repeating Data, so that cardinality if dataset becomes infinte
        ds = ds.repeat()
    # prefetching allows later images to be prepared while the current image is being processed
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def show_results(model, num_images=3, size=256):
    def plot_results(lowres, preds, highres):
        """
        Displays low resolution image and super resolution image
        """
        plt.figure(figsize=(24, 14))
        plt.subplot(131), plt.imshow(lowres), plt.title(
            f"Low resolution: {PSNR(lowres, highres).numpy():.2f}"
        )
        plt.subplot(132), plt.imshow(preds), plt.title(
            f"Prediction: {PSNR(preds, highres).numpy():.2f}"
        )
        plt.subplot(133), plt.imshow(highres), plt.title("High resolution")
        plt.show()

    for lowres, highres in val.take(num_images):
        lowres, highres = random_crop(lowres, highres, hr_crop_size=size, scale=4)
        # lowres = tf.image.random_crop(lowres, (150, 150, 3))

        start = time.time()
        preds = model(lowres[tf.newaxis, ...])[0] / 255
        print(f"Time taken to infer the image: {time.time()-start:.2f} seconds")

        # resize the lowres image to the same size as the pred using bicubic interpolation
        lowres = tf.image.resize(lowres, (size, size))

        # clip the values to be in the range of 0-255
        lowres = tf.clip_by_value(lowres, 0, 255)
        # divide by 255 to scale the pixel values to 0-1
        lowres = lowres / 255

        highres = tf.cast(highres, tf.float32) / 255

        plot_results(lowres, preds, highres)


# Download DIV2K from TF Datasets
# Using bicubic 4x degradation type
div2k_data = tfds.image.Div2k(config="bicubic_x4")
div2k_data.download_and_prepare()

# Taking train data from div2k_data object
train = div2k_data.as_dataset(split="train", as_supervised=True)
train_cache = train
# Validation data
val = div2k_data.as_dataset(split="validation", as_supervised=True)
val_cache = val

train_ds = dataset_object(train_cache, training=True)
val_ds = dataset_object(val_cache, training=True)
