import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage.color import rgb2gray
from keras.layers import Input, Convolution2D, Activation, merge
from keras.models import Model
from keras import optimizers
import sol5_utils

GRAY_SCALE = 1
RGB = 2


def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation
    :param filename: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    :return: This function returns an image, represented by a matrix of type np.float64 with intensities
    (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    """

    image = imread(filename, mode='RGB')
    im_normalized = image.astype(np.float64) / 255

    if representation == RGB:
        return im_normalized

    else:
        im_normalized_grayscale = rgb2gray(im_normalized)
        return im_normalized_grayscale


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """

    :param filenames: A list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
        and returns a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return: a Python generator object that outputs random tuples of the form (source_batch, target_batch),
        where each output variable is an array of shape (batch_size, 1, height, width).
        target_batch is made of clean images, and source_batch is their respective randomly corrupted version
        according to corruption_func(im).
    """
    cached_images = {}

    crop_height, crop_width = crop_size

    while True:
        source_batch = np.empty((batch_size, 1, crop_height, crop_width))
        target_batch = np.empty((batch_size, 1, crop_height, crop_width))

        for i in range(batch_size):
            filename = np.random.choice(filenames)

            if filename in cached_images:
                image = cached_images[filename]
            else:
                image = read_image(filename, GRAY_SCALE)
                cached_images[filename] = image

            corrupted_image = corruption_func(image)

            im_height, im_width = image.shape

            # random choice for the top left pixel pf the patch
            patch_top_y = np.random.choice(im_height - crop_height + 1)
            patch_left_x = np.random.choice(im_width - crop_width + 1)

            clean_patch = image[patch_top_y: patch_top_y + crop_height, patch_left_x: patch_left_x + crop_width]
            corrupted_patch = corrupted_image[patch_top_y: patch_top_y + crop_height, patch_left_x: patch_left_x + crop_width]

            source_batch[i, 0, :, :] = corrupted_patch - 0.5
            target_batch[i, 0, :, :] = clean_patch - 0.5

        yield (source_batch, target_batch)


def resblock(input_tensor, num_channels):
    """
    :param input_tensor: a symbolic input tensor
    :param num_channels: number of channels for each of the convolutional layers
    :return: the symbolic output tensor of the layer configuration of a residual block.
    """
    conv1 = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    relu = Activation('relu')(conv1)
    conv2 = Convolution2D(num_channels, 3, 3, border_mode='same')(relu)
    return merge([input_tensor, conv2], mode='sum')


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Returns an untrained Keras model with input dimension the shape of (1, height, width) and all convolutional layers
    (including residual blocks) with number of output channels equal to num_channels, except the very last convolutional
    layer which should have a single output channel. The number of residual blocks should be equal to num_res_blocks.
    :param height: height of input tensor
    :param width: width of input tensor
    :param num_channels: number of output channels from every inner layer.
    :param num_res_blocks:
    :return:
    """
    input_tensor = Input(shape=(1, height, width))
    conv1 = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    relu = Activation('relu')(conv1)
    block_output = resblock(relu, num_channels)  # first residual block in the model

    for i in range(num_res_blocks - 1):  # range is -1 because we created the first block outside the loop
        block_output = resblock(block_output, num_channels)

    res_blocks_output = merge([relu, block_output], mode='sum')
    output = Convolution2D(1, 3, 3, border_mode='same')(res_blocks_output)
    model = Model(input=input_tensor, output=output)
    return model


def train_model(model, images, corruption_func, batch_size, samples_per_epoch, num_epochs, num_valid_samples):
    """
    Trains a given model with given dataset.
    :param model: a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument, and
    returns a randomly corrupted version of the input image.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param samples_per_epoch: The number of samples in each epoch.
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch.
    :return:
    """
    adam = optimizers.Adam(beta_2=0.9)

    # divide the images into a training set and validation set, using an 80-20 split
    num_of_images = len(images)
    divide_idx = (8 * num_of_images) // 10
    training_set = images[:divide_idx]
    validation_set = images[divide_idx:]

    training_dataset = load_dataset(training_set, batch_size, corruption_func, model.input_shape[2:4])
    validation_dataset = load_dataset(validation_set, batch_size, corruption_func, model.input_shape[2:4])

    model.compile(loss='mean_squared_error', optimizer=adam)

    model.fit_generator(training_dataset, samples_per_epoch=samples_per_epoch,  nb_epoch=num_epochs,
                        validation_data=validation_dataset, nb_val_samples=num_valid_samples)


def restore_image(corrupted_image, base_model):
    """
    Restores the given corrupted image using the base_model neural network.
    :param corrupted_image: A grayscale image of shape (height, width) with values in the [0, 1] range of type float64
    that is affected by a corruption generated from the same corruption function encountered during training
    :param base_model: A neural network trained to restore small patches. The input and output of the network are
    images with values in range [-0.5, 0.5].
    :return: A restored image.
    """
    im_height, im_width = corrupted_image.shape

    # adjust model to new size of input
    new_input_tensor = Input(shape=(1, im_height, im_width))
    new_output_tensor = base_model(new_input_tensor)
    new_model = Model(input=new_input_tensor, output=new_output_tensor)

    corrupted_image = np.array([[corrupted_image - 0.5]])
    predicted_image = new_model.predict(corrupted_image)[0][0]
    predicted_image += 0.5
    restored_image = np.clip(predicted_image, 0, 1)
    return restored_image.astype(np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    adds gaussian noise to an image
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the
    maximal variance of the gaussian distribution.
    :return:
    """
    noisy_image = image + np.random.normal(0, np.random.uniform(min_sigma, max_sigma), image.shape)
    rounded_image = (np.round(noisy_image*255))/255
    return np.clip(rounded_image, 0, 1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    Learn denoising model
    :param num_res_blocks: number of residual blocks in the model
    :param quick_mode:
    :return:
    """
    images = sol5_utils.images_for_denoising()

    if quick_mode:
        batch_size = 10
        samples_per_epoch = 30
        num_epochs = 2
        num_valid_samples = 30

    else:
        batch_size = 100
        samples_per_epoch = 10000
        num_epochs = 5
        num_valid_samples = 1000

    model = build_nn_model(24, 24, 48, num_res_blocks)

    train_model(model, images, lambda image: add_gaussian_noise(image, 0, 0.2), batch_size, samples_per_epoch,
                num_epochs, num_valid_samples)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    adds motion blur to an image.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size: an odd integer specifying the size of the kernel
    :param angle: an angle in radians in the range [0 ,π)
    :return: The corrupted image
    """
    motion_blur_kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    blurred_image = scipy.ndimage.filters.convolve(image, motion_blur_kernel, mode='nearest')
    return np.clip(np.round(blurred_image*255)/255, 0, 1)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    adds motion blur to an image while choosing a random kernel size.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers for the kernel size.
    :return: The corrupted image
    """
    return add_motion_blur(image, np.random.choice(list_of_kernel_sizes), np.random.uniform(0, np.pi))


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """

    :param num_res_blocks:
    :param quick_mode:
    :return:
    """
    images = sol5_utils.images_for_deblurring()

    if quick_mode:
        batch_size = 10
        samples_per_epoch = 30
        num_epochs = 2
        num_valid_samples = 30

    else:
        batch_size = 100
        samples_per_epoch = 10000
        num_epochs = 10
        num_valid_samples = 1000

    model = build_nn_model(16, 16, 32, num_res_blocks)
    train_model(model, images, lambda image: random_motion_blur(image, [7]), batch_size, samples_per_epoch,
                num_epochs, num_valid_samples)
    return model


# ~~~~~~~~~~~~~~~~~~~~~~~ THE CODE USED TO CREATE THE GRAPHS:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# def depth_plot_denoise():
#
#     im = read_image("image_dataset/train/2092.jpg", 1)
#     num_blocks = [1, 2, 3, 4, 5]
#     losses = []
#
#     for num_res_blocks in num_blocks:
#         corrupted_im = add_gaussian_noise(im, 0, 0.2)
#         model = learn_denoising_model(num_res_blocks=num_res_blocks)
#         restored_im = restore_image(corrupted_im, model)
#         val_loss = model.history.history['val_loss'][-1]
#         losses.append(val_loss)
#
#     plt.plot(num_blocks, losses, 'r')
#     plt.title("Denoising")
#     plt.xlabel("num_res_blocks")
#     plt.ylabel("val_loss")
#     plt.show()
#
# depth_plot_denoise()


# def depth_plot_deblur():
#     im = read_image("text_dataset/train/0000046_orig.png", 1)
#     num_blocks = [1, 2, 3, 4, 5]
#     losses = []
#
#     for num_res_blocks in num_blocks:
#         model = learn_deblurring_model(num_res_blocks=num_res_blocks)
#         corrupted_im = random_motion_blur(im, [7])
#         restored_im = restore_image(corrupted_im, model)
#         val_loss = model.history.history['val_loss'][-1]
#         losses.append(val_loss)
#
#     plt.plot(num_blocks, losses, 'r')
#     plt.title("Deblurring")
#     plt.xlabel("num_res_blocks")
#     plt.ylabel("val_loss")
#     plt.show()
#
# depth_plot_deblur()
