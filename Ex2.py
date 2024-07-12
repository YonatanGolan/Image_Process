import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import linalg
from tqdm import tqdm


def calculate_matches(image1, image2):
    """
    The function gets two images and compute matches using "sift" algorithm
    """
    # Load the images
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Initialize the FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform matching
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches, keypoints1, keypoints2


def dlt(matrix1, matrix2):
    """
    This function implements direct linear transform, using SVD to compute the homography between the matrices
    :param matrix1: first set of 4 (x,y) coordinates from matches between images
    :param matrix2: second set of 4 (x,y) coordinates from matches between images
    :return: Homography 3x3 matrix using SVD
    """
    assert len(matrix1) == len(matrix2) == 4, "wrong number of coordinates in the matrices"

    # Creating A matrix
    A_matrix = []
    for i in range(len(matrix1)):
        xi, yi = matrix1[i]
        xi_tag, yi_tag = matrix2[i]

        A_matrix.append([-xi, -yi, -1, 0, 0, 0, xi_tag * xi, xi_tag * yi, xi_tag])
        A_matrix.append([0, 0, 0, -xi, -yi, -1, yi_tag * xi, yi_tag * yi, yi_tag])

    A_matrix = np.array(A_matrix)
    # Perform Singular Value Decomposition (SVD) on A.
    _, _, V_h = linalg.svd(A_matrix)
    # make last row of Vh 3x3 matrix
    H = V_h[-1, :].reshape((3, 3))
    # Normalize to make H[2,2] = 1
    # H /= H[2, 2]
    return H


def RANSAC(coordinates1, coordinates2, threshold, max_iterations=1000):
    """"
    This function find the best homography
    :param coordinates1: set of coordinates from the first image
    :param coordinates2: set of coordinates from the second image
    :param threshold: Threshold value
    :param max_iterations: Maximum iterations to perform homography
    :return: Best homography matrix 3X3 (which gives the maximum inliers) and the number of inliers
    """
    ret_homography = None
    ret_inliers = 0
    random.seed(43)
    # iterations will be done max_iteration times until the "best homography" will be found
    # best homography - the one that gives maximal count of inliners
    for _ in tqdm(range(max_iterations)):

        # pick 4 points from coordinates1 and coordinates2
        random_indices = random.sample(range(len(coordinates1)), 4)
        matrix1 = [coordinates1[i] for i in random_indices]
        matrix2 = [coordinates2[i] for i in random_indices]

        # Compute homography
        curr_homography = dlt(matrix1, matrix2)

        # Counting the inliers
        inliers = 0
        for i in range(len(coordinates1)):
            # To make sure the shape is correct: (1,1,2)
            point1 = np.array([[[coordinates1[i][0], coordinates1[i][1]]]], dtype=np.float32)
            trans_point = cv2.perspectiveTransform(point1, curr_homography)  # Transform point1
            # distance between the transformed point and the original point
            dist = np.linalg.norm(coordinates2[i] - trans_point[0][0])
            if dist < threshold:
                inliers += 1
        # Updating best homography and inliers by checking the current number of inliers
        if inliers > ret_inliers:
            ret_inliers = inliers
            ret_homography = curr_homography

    return ret_homography, ret_inliers


def stitch_images(image1, image2, homography):
    """
    This function gets two images and stitch them together
    """
    # Get images height and width , turn them to 1X2 vectors
    h1, w1 = cv2.imread(image1).shape[:2]
    h2, w2 = cv2.imread(image2).shape[:2]
    corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    corners2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
    # Translate the homography to take under account the field of view of the second image
    corners2_transformed = cv2.perspectiveTransform(corners2, np.linalg.inv(homography).astype(np.float32))
    all_corners = np.concatenate((corners1, corners2_transformed), axis=0)
    x, y, w, h = cv2.boundingRect(all_corners)

    # Adjust the homography matrix to map from img2 to img1
    H_adjusted = np.linalg.inv(homography)

    # Warp the images
    img1_warped = cv2.warpPerspective(cv2.imread(image1), np.eye(3), (w, h))
    img2_warped = cv2.warpPerspective(cv2.imread(image2), H_adjusted, (w, h))

    # Combine the warped images into a single output image
    output = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)
    plt.imshow(img1_warped)
    plt.figure()
    plt.imshow(img2_warped)

    # Create a mask for the overlapping region
    mask1 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask1, [np.int32(corners1)], (255))
    mask2 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask2, [np.int32(corners2_transformed)], (255))
    overlap_mask = cv2.bitwise_and(mask1, mask2)
    not_overlap_img2_mask = cv2.bitwise_and(cv2.bitwise_not(overlap_mask), mask2)

    # Blend only the overlapping region

    blended = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)

    # Copy img1_warped and img2_warped to blended using the overlap_mask
    blended = cv2.bitwise_and(blended, blended, mask=overlap_mask)
    blended += cv2.bitwise_and(img1_warped, img1_warped, mask=cv2.bitwise_not(overlap_mask))
    blended += cv2.bitwise_and(img2_warped, img2_warped, mask=not_overlap_img2_mask)

    plt.figure()
    plt.imshow(blended, cmap='gray')
    cv2.imwrite('panoramic_image.jpg', blended)


# Main

# Make sure you use the right path
image1 = './Hanging1.png'
image2 = './Hanging2.png'

# Calculate good matches between the images and obtain keypoints
matches, keypoints1, keypoints2 = calculate_matches(image1, image2)

# Extract coordinates of the keypoints
coordinates1 = np.float32([keypoints1[match.queryIdx].pt for match in matches])
coordinates2 = np.float32([keypoints2[match.trainIdx].pt for match in matches])

# RANSAC to find the best homography
homography, inliers = RANSAC(coordinates1, coordinates2, threshold=10000)

# Stitch the images together
stitch_images(image1, image2, homography)
