import numpy as np
import math
import scipy.signal
from scipy.misc import imread
from skimage.color import rgb2gray


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
    im_normalized = (image / 255).astype(np.float64)  # normalize the pixels to range [0, 1]

    if representation == 2:
        return im_normalized

    else:
        im_normalized_grayscale = rgb2gray(im_normalized)
        return im_normalized_grayscale


def create_base_matrix(matrix_size):
    x_index_matrix, y_index_matrix = np.meshgrid(np.arange(matrix_size), np.arange(matrix_size))
    indices_multi_matrix = np.multiply(x_index_matrix, y_index_matrix)
    fourier_coefficient_base = math.e ** (-(2 * math.pi * 1j) / matrix_size)
    return fourier_coefficient_base ** indices_multi_matrix


def create_inv_base_matrix(matrix_size):
    y_index_matrix, x_index_matrix = np.meshgrid(np.arange(matrix_size),
                                                 np.arange(matrix_size))
    indices_multi_matrix = np.multiply(y_index_matrix, x_index_matrix)
    inv_fourier_coefficient_base = math.e ** ((2 * math.pi * 1j) / matrix_size)
    return (1/matrix_size)*(inv_fourier_coefficient_base ** indices_multi_matrix)


def DFT(signal):
    """
    Returns the fourier transform of a 1D array
    :param signal: array of dtype float64 with shape (N,1)
    :return: The complex Fourier signal
    """
    signal = np.array(signal, dtype=np.complex128)
    fourier_coefficients_matrix = create_base_matrix(signal.size)
    DFT = np.dot(fourier_coefficients_matrix, signal)
    DFT.shape = (DFT.size, 1)
    return DFT


def IDFT(fourier_signal):
    """
    Returns the inverse fourier transform of a signal
    :param fourier_signal: array of dtype complex128 with shape (N, 1)
    :return: The complex inverse signal
    """
    fourier_signal = np.array(fourier_signal)
    inv_fourier_coefficients_matrix = create_inv_base_matrix(fourier_signal.size)
    IDFT = np.dot(inv_fourier_coefficients_matrix, fourier_signal)
    IDFT.shape = (IDFT.size, 1)
    return IDFT


def DFT2(image):
    """
    Returns the fourier transform of an image
    :param image: a grayscale image of dtype float64
    :return: an image of dtype complex128
    """
    m_matrix = create_base_matrix(len(image))
    n_matrix = create_base_matrix(len(image[0]))
    return np.dot(m_matrix, np.dot(image, n_matrix))


def IDFT2(fourier_image):
    """
    Returns the inverse fourier transform of an image (2D array)
    :param fourier_image: a 2D array of dtype complex128
    :return: an image of dtype complex128
    """
    m_matrix = create_inv_base_matrix(len(fourier_image))
    n_matrix = create_inv_base_matrix(len(fourier_image[0]))
    return np.dot(m_matrix, np.dot(fourier_image, n_matrix))


def conv_der(im):
    """
    Computes the magnitude of an image derivative using convolution.
    :param im: A grayscale image of type float64
    :return: The magnitude of the image derivative, with the same dtype and shape as the input.
    """
    der_x = np.array([[1, 0, -1]])
    der_y = der_x.T
    der_x_image = scipy.signal.convolve2d(im, der_x, mode="same")
    der_y_image = scipy.signal.convolve2d(im, der_y, mode="same")
    magnitude_im = np.sqrt(np.abs(der_x_image)**2 + np.abs(der_y_image)**2)
    return magnitude_im


def fourier_der(im):
    """
    Computes the magnitude of an image derivative using Fourier transform.
    :param im: A grayscale image of type float64
    :return: The magnitude of the image derivative, with the same dtype and shape as the input.
    """
    fourier_der_im = DFT2(im)
    fourier_der_im = np.fft.fftshift(fourier_der_im)
    v_index_matrix, u_index_matrix = np.meshgrid(np.arange(-(len(im[0])/2), (len(im[0])/2)),
                                                 np.arange(-(len(im)/2), (len(im)/2)))

    fourier_der_x = np.multiply(fourier_der_im, u_index_matrix)
    fourier_der_y = np.multiply(fourier_der_im, v_index_matrix)
    der_x = ((2*math.pi*1j)/len(im))*IDFT2(fourier_der_x)
    der_y = ((2*math.pi*1j)/len(im[0]))*IDFT2(fourier_der_y)
    magnitude_im = np.sqrt(np.abs(der_x)**2 + np.abs(der_y)**2)
    return magnitude_im


def create_kernel(kernel_size):
    """
    Creates a blurring kernel of
    :param kernel_size: The size of the gaussian kernel in each dimension
    :return: a blurring kernel with the shape kernel_size X kernel_size containing the approximation
    of the gaussian distribution using the binomial coefficients
    """
    kernel_base = np.array([[1, 1]]).astype(np.float64)
    for i in range(kernel_size - 2):
        kernel_base = scipy.signal.convolve2d(kernel_base, np.array([[1, 1]]))
    kernel = np.outer(kernel_base, kernel_base)
    norm_kernel = kernel/np.sum(kernel)
    return norm_kernel


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


def blur_fourier(im, kernel_size):
    """
    Blurs the given image using pointwise multiplication between the fourier transform of the blurring gaussian kernel
    and between the fourier transform of the image.
    :param im: The given image to blur (grayscale float64 image).
    :param kernel_size: The size of the gaussian kernel in each dimension
    :return: The output blurry image (grayscale float64 image).
    """
    kernel = create_kernel(kernel_size)
    zeros = np.zeros_like(im)

    zero_height, zero_width = im.shape
    ker_height, ker_width = kernel.shape
    zeros[math.floor(zero_height/2) - math.floor(ker_height/2) - 1: math.floor(zero_height/2) + math.floor(ker_height/2),
          math.floor(zero_width/2) - math.floor(ker_width/2) - 1: math.floor(zero_width/2) + math.floor(ker_width/2)] = kernel

    fourier_kernel_big = DFT2(np.fft.ifftshift(zeros))
    fourier_im = DFT2(im)
    blurred_fourier_im = np.multiply(fourier_im, fourier_kernel_big)
    blurred_im = np.abs(IDFT2(blurred_fourier_im))
    return blurred_im
