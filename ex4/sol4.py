import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
from scipy.misc import imsave
import shutil
import random
import math

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass

import sol4_utils

CORNER_DETECTOR_K = 0.04


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max()*0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num)+1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def harris_corner_detector(im):
    """
    Detects harris corners.
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    x_der = np.array([[1, 0, -1]])
    y_der = np.array([[1, 0, -1]]).T

    im_x = scipy.signal.convolve2d(im, x_der, mode='same', boundary='symm')
    im_y = scipy.signal.convolve2d(im, y_der, mode='same',  boundary='symm')

    im_xx = sol4_utils.blur_spatial(im_x*im_x, 3)
    im_yy = sol4_utils.blur_spatial(im_y*im_y, 3)
    im_xy = sol4_utils.blur_spatial(im_x*im_y, 3)

    im_det = im_xx*im_yy - im_xy**2
    im_trace = im_xx + im_yy
    R = im_det - CORNER_DETECTOR_K*(im_trace**2)
    local_maximums = non_maximum_suppression(R)
    y_indices, x_indices = np.where(local_maximums)
    return np.column_stack((x_indices, y_indices))


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1]-radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0]-radius))
    ret = corners[legit, :]
    return ret


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    K = 1 + 2*desc_rad
    N = len(pos)
    descriptors = np.zeros((N, K, K))
    for i in range(N):
        xl3, yl3 = pos[i, :]
        x_neighborhood = np.arange(xl3 - desc_rad, xl3 + desc_rad + 1)
        y_neighborhood = np.arange(yl3 - desc_rad, yl3 + desc_rad + 1)
        x_neighborhood_coords = np.tile(x_neighborhood, K)
        y_neighborhood_coords = np.repeat(y_neighborhood, K)
        sampled_descriptor = scipy.ndimage.map_coordinates(
            im, [y_neighborhood_coords, x_neighborhood_coords], order=1, prefilter=False)
        mean = np.mean(sampled_descriptor)
        if not np.all(sampled_descriptor - mean):  # if norm is 0
            descriptor = np.zeros((K, K))
        else:
            descriptor = (sampled_descriptor - mean) / (np.linalg.norm(sampled_descriptor - mean))
        descriptors[i, :, :] = descriptor.reshape(K, K)
    return descriptors


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    corners = spread_out_corners(pyr[0], 7, 7, 20)  # corners coordinates in original image

    # corners coordinates transformed to the third pyramid level
    descriptors = sample_descriptor(pyr[2], (2**-2)*corners, 3)
    return [corners, descriptors]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """

    desc1_flattened = desc1.reshape(len(desc1), len(desc1[0])**2)
    desc2_flattened = desc2.reshape(len(desc2), len(desc2[0])**2)

    # contains the score for every to descriptors. s[i,j] = match score between the ith descriptor
    # from desc1 and the jth descriptor from desc2.
    scores_matrix = np.dot(desc1_flattened, desc2_flattened.T)

    max_rows_idxs = np.argpartition(scores_matrix, len(desc2) - 2)[:, -2:]
    max_columns_idxs = np.argpartition(scores_matrix.T, len(desc1) - 2)[:, -2:]

    max_rows = np.zeros_like(scores_matrix)
    max_columns = np.zeros_like(scores_matrix.T)

    max_rows[np.arange(len(desc1)), max_rows_idxs[:, 0]] = 1
    max_rows[np.arange(len(desc1)), max_rows_idxs[:, 1]] = 1

    max_columns[np.arange(len(desc2)), max_columns_idxs[:, 0]] = 1
    max_columns[np.arange(len(desc2)), max_columns_idxs[:, 1]] = 1

    greater_than_min = np.array(scores_matrix > min_score).astype(np.int)

    matching_desc = np.multiply(np.multiply(max_rows, max_columns.T), greater_than_min)
    match_ind1, match_ind2 = np.where(matching_desc)
    return [match_ind1, match_ind2]


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    ones = np.ones((len(pos1), 1))
    expanded_pos1 = np.append(pos1, ones, axis=1)
    expanded_pos2 = np.dot(expanded_pos1, H12.T)
    z_vec = expanded_pos2[:, 2].reshape(len(pos1), 1)
    pos2 = expanded_pos2[:, : 2] / z_vec
    return pos2


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    best_inliers = np.array([])
    max_num_inliers = 0
    for i in range(num_iter):
        j, k = random.sample(range(len(points1)), 2)

        sample_points1 = np.array([points1[j, :], points1[k, :]])
        sample_points2 = np.array([points2[j, :], points2[k, :]])

        homography_matrix = estimate_rigid_transform(sample_points1, sample_points2, translation_only)

        points1_transformed = apply_homography(points1, homography_matrix)

        squared_distance = np.linalg.norm((points1_transformed - points2), axis=1)**2

        in_range_of_tol = (squared_distance < inlier_tol).astype(np.int)

        if np.sum(in_range_of_tol) > max_num_inliers:
            max_num_inliers = np.sum(in_range_of_tol)
            best_inliers = np.nonzero(in_range_of_tol)[0].T

    homography_matrix = estimate_rigid_transform(points1[best_inliers, :], points2[best_inliers, :], translation_only)
    homography_matrix /= homography_matrix[2, 2]

    return [homography_matrix, best_inliers]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :param points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    stacked_images = np.hstack((im1, im2))
    plt.imshow(stacked_images, cmap=plt.cm.gray)
    im1_width = len(im1[0])
    inliers_pointer = 0

    for i in range(len(points1)):
        # adapt x,y coordinates of points in im2 to x,y coordinates in stacked_image
        x_coordinates = np.array([points1[i, 0], (points2[i, 0] + im1_width)])
        y_coordinates = np.array([points1[i, 1], points2[i, 1]])

        # ith pair is inliers
        if i == inliers[inliers_pointer]:
            if inliers_pointer < len(inliers) - 1:
                inliers_pointer += 1
            plt.plot(x_coordinates, y_coordinates, mfc='r', c='y', lw=.4, ms=3, marker='o')

        # ith pair is outliers
        else:
            plt.plot(x_coordinates, y_coordinates, mfc='r', c='b', lw=.2, ms=3, marker='o')

    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    H2m = [0]*(len(H_succesive) + 1)
    H2m[m] = np.eye(3)
    for i in range(len(H_succesive)//2 + 1):
        if (m-1-i) >= 0:
            H2m[m-1-i] = np.dot(H2m[m-i], H_succesive[m-1-i])
            H2m[m-1-i] /= H2m[m-1-i][2, 2]  # normalize
        if (m+1+i) < len(H2m):
            if not i:  # first iteration i==0
                H2m[m+1] = np.linalg.inv(H_succesive[m])
                H2m[m+1] /= H2m[m+1][2, 2]  # normalize
            else:
                H2m[m+1+i] = np.dot(H2m[m+i], np.linalg.inv(H_succesive[m+i]))
                H2m[m+1+i] /= H2m[m+1+i][2, 2]  # normalize
    return H2m


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the upper left corner,
     and the second row is the [x,y] of the lower right corner
    """
    image_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
    transformed_corners = apply_homography(image_corners, homography)
    min_x = np.amin(transformed_corners[:, 0])
    min_y = np.amin(transformed_corners[:, 1])
    max_x = np.amax(transformed_corners[:, 0])
    max_y = np.amax(transformed_corners[:, 1])
    return np.array([[math.floor(min_x), math.floor(min_y)],
                     [math.ceil(max_x), math.ceil(max_y)]])


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    height, width = image.shape
    upper_left, lower_right = compute_bounding_box(homography, width, height)
    min_x, min_y = upper_left
    max_x, max_y = lower_right

    x_coordinates_matrix, y_coordinates_matrix = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
    h, w = x_coordinates_matrix.shape

    x_coordinates_array = x_coordinates_matrix.reshape((h*w, 1))
    y_coordinates_array = y_coordinates_matrix.reshape((h*w, 1))

    all_points = np.hstack((x_coordinates_array, y_coordinates_array))
    all_points_back_warped = apply_homography(all_points, np.linalg.inv(homography))

    x_coords_back_warped = all_points_back_warped[:, 0]
    y_coords_back_warped = all_points_back_warped[:, 1]

    points_warped = scipy.ndimage.map_coordinates(image, [y_coords_back_warped, x_coords_back_warped],
                                                  order=1, prefilter=False)

    return points_warped.reshape((h, w))


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 200, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle
        # image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from
        # which the panorama will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
