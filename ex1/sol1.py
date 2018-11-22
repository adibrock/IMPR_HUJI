import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage.color import rgb2gray

CONVERSION_MATRIX_RGB_TO_YIQ = np.array([[0.299, 0.587, 0.114],
                                        [0.596, -0.275, -0.321],
                                        [0.212, -0.523, 0.311]])

CONVERSION_MATRIX_YIQ_TO_RGB = np.linalg.inv(CONVERSION_MATRIX_RGB_TO_YIQ)


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
    image = image.astype(np.float64)  # change the pixels type to float64
    im_normalized = (image / 255).astype(np.float64)  # normalize the pixels to range [0, 1]

    if representation == 2:
        return im_normalized

    else:
        im_normalized_grayscale = rgb2gray(im_normalized)
        return im_normalized_grayscale


def imdisplay(filename, representation):
    """
    Reads an image file and converts it into a given representation, than displays it.
    :param filename: string containing the image filename to display.
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    :return: This function displays an image, represented by a matrix of type np.float64 with intensities
    (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    """
    image = read_image(filename, representation)

    if representation == 2:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()


def rgb2yiq(imRGB):
    """
    Converts an RGB image to YIQ color space.
    :param imRGB: height × width × 3 np.float64 matrix with values in [0, 1], representing an RGB image.
    :return: A matrix representing the image in the YIQ color space with float64 values in [0, 1].
    """
    return np.dot(imRGB, CONVERSION_MATRIX_RGB_TO_YIQ)


def yiq2rgb(imYIQ):
    """
    Converts an image matrix from YIQ color space to RGB image.
    :param imYIQ: height × width × 3 np.float64 matrix with values in [0, 1], representing an image in YIQ color space.
    :return: An RGB image with float64 values in [0, 1]
    """
    return np.dot(imYIQ, CONVERSION_MATRIX_YIQ_TO_RGB)


def histogram_equalize(im_orig):
    """
    performs histogram equalization on a given grayscale ro RGB image.
    :param im_orig: the input grayscale or RGB float64 image with values in [0, 1].
    :return: returns a list [im_eq, hist_orig, hist_eq] where:
            * im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
            * hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
            * hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    im_orig = np.array(im_orig)

    if len(im_orig.shape) != 2:  # RGB image
        yiqImg = rgb2yiq(im_orig)
        y_channel = yiqImg[:, :, 0]

        eq_y_channel, y_channel_hist_orig, y_channel_hist_eq = grayscale_image_equalize(y_channel)

        # put equalized Y channel back to YIQ image
        yiqImg[:, :, 0] = eq_y_channel

        # convert equalized YIQ image back to RGB and clips values to be between [0, 1]
        eq_rgb_im = np.clip(yiq2rgb(yiqImg), 0, 1)
        return [eq_rgb_im, y_channel_hist_orig, y_channel_hist_eq]

    else:
        return grayscale_image_equalize(im_orig)


def grayscale_image_equalize(im):
    """
    performs the actual histogram equalization on the given array
    :param im: original image in case of a grayscale image, or the Y channel of the YIQ image
               in case of an RGB image.
    :return: equalized grayscale image or Y channel, the original histogram of the input,
             and the equalized histogram of the input.
    """
    # clip values to get ints only between 0 and 255
    img_orig_int = np.clip(np.array(im * 255), 0, 255).astype(np.uint8)
    orig_hist, bins = np.histogram(img_orig_int, 256, [0, 256])

    cum_hist = np.cumsum(orig_hist)

    # finds the lowest index where the cumulative histogram is not zero
    lowest_gray_level = np.where(cum_hist != 0)[0][0]

    # equalize cumulative histogram
    eq_cum_hist = np.round((cum_hist - cum_hist[lowest_gray_level]) /
                           (cum_hist[255] - cum_hist[lowest_gray_level]) * 255)

    equalized_im = eq_cum_hist[img_orig_int]

    equalized_hist, bins = np.histogram(equalized_im, 256, [0, 256])

    # converts image back to float64 values in [0, 1]
    equalized_im = (equalized_im/255).astype(np.float64)

    return [equalized_im, orig_hist, equalized_hist]


def quantize(im_orig, n_quant, n_iter):
    """
    performs optimal quantization of a given grayscale or RGB image
    :param im_orig: input grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: the number of intensities the output im_quant image should have.
    :param n_iter: the maximum number of iterations of the optimization procedure.
    :return: a list [im_quant, error] where :
            * im_quant - is the quantized output image.
            * error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
              quantization procedure.
    """
    if len(im_orig.shape) != 2:  # RGB image
        yiqImg = rgb2yiq(im_orig)
        y_channel = yiqImg[:, :, 0]

        seg_divisors, seg_gray_levels, iteration_errors = perform_quantization_iterations(y_channel, n_quant, n_iter)
        quantized_y = quantize_image(y_channel, seg_divisors, seg_gray_levels, n_quant)
        # put equalized Y channel back to YIQ image
        yiqImg[:, :, 0] = quantized_y

        # convert equalized YIQ image back to RGB and clips values to be between [0, 1]
        quant_rgb_im = np.clip(yiq2rgb(yiqImg), 0, 1)

        return [quant_rgb_im, iteration_errors]

    else:
        seg_divisors, seg_gray_levels, iteration_errors = \
            perform_quantization_iterations(im_orig, n_quant, n_iter)

        return [quantize_image(im_orig, seg_divisors, seg_gray_levels, n_quant), iteration_errors]


def perform_quantization_iterations(gray_img, n_quant, n_iter):
    """
    Performs iterations to calculate the optimal gray levels and segment divisors for the image
    :param gray_img: the original image in case of grayscale image or Y channel in case of RGB
    :param n_quant: the number of intensities the output im_quant image should have.
    :param n_iter: the maximum number of iterations of the optimization procedure.
    :return: a list of [seg_divisors, seg_gray_levels, iteration_errors] where:
             * seg_divisors - a list of the segment divisors (z's) found in the optimization procedure
             * seg_gray_levels - a list of the gray levels found in the optimization procedure that
              each segment should be mapped to
             * iteration_errors - a list of the total intensities error from each iteration.
    """
    img_orig_int = np.clip((gray_img * 255).astype(np.uint8), 0, 255)

    histogram, bins = np.histogram(img_orig_int, 256, [0, 256])

    seg_gray_levels = np.zeros(n_quant, dtype=np.int64)
    iteration_errors = np.zeros(n_iter)

    seg_divisors = set_initial_division_of_segments(n_quant, histogram)

    for i in range(n_iter):
        iter_error = 0
        for j in range(n_quant):
            seg_gray_levels[j] = (calculate_new_segment_gray_level(seg_divisors[j],
                                                                   seg_divisors[j+1],
                                                                   histogram))
            if j == 0:
                iter_error += calculate_segment_error(seg_divisors[j],
                                                      seg_divisors[j+1],
                                                      histogram,
                                                      seg_gray_levels[j])
                continue

            seg_divisors[j] = (seg_gray_levels[j-1] + seg_gray_levels[j]) / 2

            iter_error += calculate_segment_error(seg_divisors[j],
                                                  seg_divisors[j+1],
                                                  histogram,
                                                  seg_gray_levels[j])

        iteration_errors[i] = iter_error
        if i > 0 and iter_error == iteration_errors[i - 1]:
            break

    return [seg_divisors, seg_gray_levels, iteration_errors]


def set_initial_division_of_segments(n_quant, histogram):
    """
    Creates a list for the segment divisors and sets the initial segment divisors in order to start
    the optimization procedure, so that in each segment there are approximately the same number of pixels.
    :param n_quant: the number of intensities the output im_quant image should have.
    :param histogram: the histogram of the image (or Y channel)
    :return: the list of segment divisors to work with.
    """
    seg_divisors = np.zeros(n_quant + 1, dtype=np.int64)
    seg_divisors[-1] = 255

    cum_hist = np.cumsum(histogram)
    pix_in_segment = (cum_hist[-1] / n_quant).astype(int)

    for i in range(n_quant):
        index = np.searchsorted(cum_hist, i*pix_in_segment)
        seg_divisors[i] = index

    return seg_divisors


def calculate_new_segment_division(q1, q2):
    """
    Calculates the new segment divisors based on the gray levels of the segments
    :param q1: gray level the segment before is mapped to
    :param q2: gray level the segment after is mapped to
    :return: the new segment divisor
    """
    return (q1 + q2) // 2


def calculate_new_segment_gray_level(z1, z2, histogram):
    """
    Calculates the new gray levels for the current segment based on the segment divisors
    :param z1: beginning index of cur segment
    :param z2: end index of cur segment
    :param histogram: histogram of the image
    :return: the new gray level the current segment is mapped to.
    """
    hist_part = histogram[z1: z2 + 1]
    gray_level = (np.dot(hist_part, (np.arange(z1, z2 + 1))) / np.sum(hist_part)).astype(int)
    return gray_level


def calculate_segment_error(z1, z2, histogram, gray_level):
    """
    Calculates the sum of the intensities error in a segment
    :param z1: beginning index of segment
    :param z2: end index of segment
    :param histogram: histogram of the image
    :param gray_level: the gray level teh current segment is mapped to
    :return: sum of all intensities error in cur segment
    """
    qi_array = np.ones(z2+1 - z1) * gray_level
    error = (qi_array - np.arange(z1, z2 + 1))
    seg_error = np.dot(histogram[z1: z2 + 1], np.power(error, 2))
    return seg_error


def quantize_image(img, divisors_array, gray_levels_array, n_quant):
    """
    Performs the actual quantization of the image - maps every pixel to the right grayscale
    :param img: the image to quantize (actual image if grayscale, Y channel if RGB)
    :param divisors_array: final array of segment divisors
    :param gray_levels_array: final array of gray level for each segment to be mapped to
    :param n_quant: the number of intensities the output im_quant image should have.
    :return: the quantized image
    """

    new_img = np.zeros_like(img).astype(float)

    for i in range(n_quant):
        all_bigger_than_zi = img >= divisors_array[i] / 256
        all_smaller_than_zi = img <= divisors_array[i+1] / 256
        new_img[np.logical_and(all_bigger_than_zi, all_smaller_than_zi)] = gray_levels_array[i] / 256

    return new_img
