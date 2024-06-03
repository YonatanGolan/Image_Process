import numpy as np
import cv2
from matplotlib import pyplot as plt


def interpulation(pic):  # 1.a
    """ This function does super resolution by bilinear interpulation
        pic: 2D numpy array of the original image (n*m)
        return: 2D numpy array of the new image (2n*2m)"""
    n, m = pic.shape
    new_pic = np.zeros((2 * n, 2 * m))
    for i in range(n):
        for j in range(m):
            new_pic[2 * i, 2 * j] = pic[i, j]
            new_pic[2 * i + 1, 2 * j] = pic[i, j]
            new_pic[2 * i, 2 * j + 1] = pic[i, j]
            new_pic[2 * i + 1, 2 * j + 1] = pic[i, j]
    return new_pic


def plot_hist(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])  # allowed?
    plt.plot(hist, color='gray')
    plt.xlim([0, 256])
    plt.show()


def stretch_contrast(img):
    """
    This function stretches the contrast of the image - for each pixel value, the new value is calculated by:
    new_value = (value - min_value) * 255 / (max_value - min_value)
    :param img: 2D numpy array of the original image
    :return: stretched image
    """
    min_value = np.min(img)
    max_value = np.max(img)
    new_img = (img - min_value) * 255 / (max_value - min_value)

    # Plot the new image and its histogram
    plt.imshow(new_img, cmap='gray')
    plt.show()
    plot_hist(new_img)
    return new_img


def equalize_hist(img):
    """
    This function equalizes the histogram of the image
    :param img: 2D numpy array of the original image
    :return: equalized image
    """
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    new_img = cdf[img]
    # Plot the new image and its histogram
    plt.imshow(new_img, cmap='gray')
    plt.show()
    plot_hist(new_img)
    return new_img


if __name__ == '__main__':
    # Part A
    # 1.Interpolation:
    # 1.b
    print('Loading peppers image')
    peppers = cv2.imread('./hw12024_input_img/hw12024/peppers.jpg', 0)
    print('Interpolating peppers image by 2')
    peppers_interp_by_2 = interpulation(peppers)
    print('Saving interpolated peppers image (by 2)')
    cv2.imwrite('./output/peppers_interp_by_2.jpg', peppers_interp_by_2)
    # 1.c
    print('Interpolating peppers image by 8 - 3 times interpulation by 2')
    peppers_interp_by_8 = interpulation(interpulation(interpulation(peppers)))
    print('Saving interpolated peppers image (by 8)')
    cv2.imwrite('./output/peppers_interp_by_8.jpg', peppers_interp_by_8)

    # 2.Equal Histogram:
    print('Loading leaf image')
    leaf = cv2.imread('./hw12024_input_img/hw12024/leafs.jpg', 0)
    # 2.a
    print('Calculating and plotting the histogram of the leaf image by the pixels values (0-255)')
    plot_hist(leaf)
    # 2.b
    print('Stretching contrast of the leaf image')
    stretch_contrast(leaf)
    # 2.c
    print('Equalizing the histogram of the leaf image')
    equalize_hist(leaf)
