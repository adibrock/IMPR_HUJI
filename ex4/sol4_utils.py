import numpy as np
from skimage.color import rgb2gray
from scipy.misc import imread
import scipy.signal


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
    image = image.astype(np.float64)
    im_normalized = (image / 255)

    if representation == 2:
        return im_normalized

    else:
        im_normalized_grayscale = rgb2gray(im_normalized)
        return im_normalized_grayscale


def create_kernel(kernel_size):
    """
    Creates a blurring kernel of
    :param kernel_size: The size of the gaussian kernel in each dimension
    :return: a blurring kernel with the shape kernel_size X kernel_size containing the approximation of the gaussian
    distribution using the binomial coefficients
    """
    kernel_base = np.array([[1, 1]]).astype(np.float64)
    for i in range(kernel_size - 2):
        kernel_base = scipy.signal.convolve2d(kernel_base, np.array([[1, 1]]))
    kernel = np.outer(kernel_base, kernel_base)
    norm_kernel = kernel/np.sum(kernel)
    return norm_kernel


def create_blur_filter(filter_size):
    """
    Creates a blurring filter
    :param filter_size: The size of the gaussian blurring filter in each dimension
    :return: a blurring filter with the shape filter_size X filter_size containing the approximation
    of the gaussian distribution using the binomial coefficients
    """
    filter_base = np.array([[1, 1]])
    for i in range(filter_size - 2):
        filter_base = scipy.signal.convolve2d(filter_base, np.array([[1, 1]]))
    norm_filter = filter_base / np.sum(filter_base)
    return norm_filter


def reduce_image(im, blurring_filter):
    blurred_im = scipy.ndimage.filters.convolve(im, blurring_filter, mode='mirror')
    blurred_im = scipy.ndimage.filters.convolve(blurred_im, blurring_filter.T, mode='mirror')
    reduced_image = blurred_im[::2, ::2]
    return reduced_image


def expand_image(im, blurring_filter):
    blurring_filter = 2*blurring_filter  # maintains constant brightness
    im_height, im_width = im.shape
    if im_height % 2 != 0:
        im_height -= 1
    if im_width % 2 != 0:
        im_width -= 1

    expanded_im = np.zeros((im_height*2, im_width*2))
    expanded_im[::2, ::2] = im
    blurred_expanded_im = scipy.ndimage.filters.convolve(expanded_im, blurring_filter,
                                                         mode='mirror')
    blurred_expanded_im = scipy.ndimage.filters.convolve(blurred_expanded_im, blurring_filter.T,
                                                         mode='mirror')
    return blurred_expanded_im


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Creates a gaussian pyramid of an image.
    :param im: a  grayscale  image  with  double  values  in  [0, 1]
    :param max_levels:  the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a
    squared filter) to be used in constructing the pyramid filter.
    :return: pyr - a standard  python  array with maximum length of max_levels, where each
                   element of the array is a grayscale image in a different level.
             filter_vec - the vector used to create the blurring filter
    """
    pyr = [im]
    blur_filter = create_blur_filter(filter_size)
    if len(np.array(im).shape) == 2:  # grayscale image
        for i in range(max_levels - 1):

            # lowest resolution image in the pyramid is not smaller than 16
            im_height, im_width = im.shape
            if im_height == 16:
                break
            if im_width == 16:
                break

            reduced_image = reduce_image(im, blur_filter)
            pyr.append(reduced_image)
            im = reduced_image

    else:  # RGB
        for i in range(max_levels - 1):
            cur_height, cur_width, rgb = im.shape

            # create an empty array half the size of the current image to save the reduced image.
            if cur_width % 2 != 0:
                cur_width += 1
            if cur_height % 2 != 0:
                cur_height += 1
            new_im = np.empty((cur_height//2, cur_width//2, 3), dtype=np.float64)

            for j in range(3):  # loop over RGB channels
                partial_color = im[:, :, j]
                reduced_color = reduce_image(partial_color, blur_filter)
                new_im[:, :, j] = reduced_color
            pyr.append(new_im)
            im = new_im
    return pyr, blur_filter


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Creates a laplacian pyramid of an image
    :param im: a  grayscale  image  with  double  values  in  [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a
    squared filter) to be used in constructing the pyramid filter.
    :return: pyr - a standard  python  array with maximum length of max_levels, where each
                   element of the array is a grayscale image in a different level.
             filter_vec - the vector used to create the blurring filter.

    """
    reduced_pyr, blur_filter = build_gaussian_pyramid(im, max_levels, filter_size)
    expanded_pyr = []
    for i in range(len(reduced_pyr) - 1):
        expanded_im = expand_image(reduced_pyr[i + 1], blur_filter)
        expanded_pyr.append(expanded_im)

    laplacian_pyr = [reduced_pyr[i] - expanded_pyr[i] for i in range(len(reduced_pyr) - 1)]
    laplacian_pyr.append(reduced_pyr[-1])
    return laplacian_pyr, blur_filter


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Reconstructs the image from it's Laplacian pyramid
    :param lpyr: the laplacian pyramid as generated in the previous function.
    :param filter_vec: the normalized vector used as the blurring filter.
    :param coeff: a vector with same size as the number of levels in the laplacian pyramid, that
    holds the corresponding coefficients for each level of the pyramid for the reconstruction of the
    image.
    :return: the reconstructed image from the pyramid, according to the given coefficients.
    """
    cur_level = len(coeff) - 1
    first_level = coeff[-1] * lpyr[-1]
    reconstructed_im = expand_image(first_level, filter_vec)
    for i in range(len(coeff) - 2):
        cur_level -= 1
        weighted_part = coeff[cur_level] * lpyr[cur_level]
        reconstructed_im = expand_image(reconstructed_im + weighted_part, filter_vec)
    return reconstructed_im + lpyr[0]


def blur_spatial(im, kernel_size):
    """
    Blurs the given image using convolution with a blurring gaussian kernel
    :param im: The given image to blur (grayscale float64 image).
    :param kernel_size: The size of the gaussian kernel in each dimension
    :return: The output blurry image (grayscale float64 image).
    """
    if kernel_size == 1:
        return im
    kernel = create_kernel(kernel_size)
    blurred_im = scipy.signal.convolve2d(im, kernel, mode='same', boundary='wrap')
    return blurred_im


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Implements image merging using pyramid blending,
    :param im1: first input grayscale image to be blended.
    :param im2: second input grayscale image t be blended.
    :param mask: a boolean (dtype == np.bool) mask containing True and False
    representing which parts of im1 and im2 should appear in the resulting im_blend
    :param max_levels: the maximal number of levels to use when generating the Gaussian
                       and Laplacian pyramids.
    :param filter_size_im: is the size of the Gaussian filter to be used in the
                           construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: is the size of the Gaussian filter to be used in the
                             construction of the Gaussian pyramid of mask.
    :return: the blended image of im1 and im2.
    """
    Lpyr_im1, blur_filter = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    Lpyr_im2, blur_filter = build_laplacian_pyramid(im2, max_levels, filter_size_im)

    mask_float = mask.astype(np.float64)
    Gpyr_mask, blur_filter = build_gaussian_pyramid(mask_float, max_levels, filter_size_mask)

    merged_lpyr = np.zeros_like(np.array(Lpyr_im1))

    levels = len(Lpyr_im1)
    for k in range(levels):
        merged_lpyr[k] = Gpyr_mask[k]*Lpyr_im1[k] + (1 - Gpyr_mask[k])*Lpyr_im2[k]

    blended_merged_image = laplacian_to_image(merged_lpyr, blur_filter, np.ones(levels))
    return np.clip(blended_merged_image, 0, 255)
