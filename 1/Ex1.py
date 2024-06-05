import numpy as np
import cv2
from matplotlib import pyplot as plt


def interpulation(img):  # 1.a
    """ This function does super resolution by bilinear interpulation
        img: 2D numpy array of the original image (n*m)
        return: 2D numpy array of the new image (2n*2m)"""
    n, m = img.shape
    new_img = np.zeros((2 * n, 2 * m))
    for i in range(n):
        for j in range(m):
            new_img[2 * i, 2 * j] = img[i, j]
            new_img[2 * i + 1, 2 * j] = img[i, j]
            new_img[2 * i, 2 * j + 1] = img[i, j]
            new_img[2 * i + 1, 2 * j + 1] = img[i, j]
    return new_img


def plot_hist(img, title='Image Histogram'):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    plt.plot(hist, color='gray')
    plt.xlim([0, 256])
    plt.xlabel('Intensity')
    plt.ylabel('Number of pixels')
    plt.title(title)
    plt.savefig('./' + title + '.jpg')  # Save the histogram - optional
    plt.show()


def stretch_contrast(img, title='Stretched Image'):
    """
    This function stretches the contrast of the image - for each pixel value, the new value is calculated by:
    new_value = (value - min_value) * 255 / (max_value - min_value)
    :param img: 2D numpy array of the original image
    :param title: title of the plot
    :return: stretched image
    """
    min_value = np.min(img)
    max_value = np.max(img)
    new_img = np.round(((img - min_value) / (max_value - min_value)) * 255).astype(np.uint8)
    # Plot the new image and its histogram
    plt.imshow(new_img, cmap='gray')
    plt.title(title)
    cv2.imwrite('./' + title + '.jpg', new_img)  # Save the image - optional
    plt.show()
    return new_img


def equalize_hist(img, title='Equalized Image'):
    """
    This function equalizes the histogram of the image
    :param img: 2D numpy array of the original image
    :param title: title of the plot
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
    plt.title(title)
    cv2.imwrite('./' + title + '.jpg', new_img)  # Save the image - optional
    plt.show()
    return new_img


if __name__ == '__main__':  # Part A
    # 1.Interpolation:
    # 1.b
    print('Loading peppers image')
    peppers = cv2.imread('./peppers.jpg', 0)
    print('Interpolating peppers image by 2')
    peppers_interp_by_2 = interpulation(peppers)
    print('Saving interpolated peppers image (by 2)')
    cv2.imwrite('./Peppers_interpolated_by_2.jpg', peppers_interp_by_2)
    # 1.c
    print('Interpolating peppers image by 8 - 3 times interpulation by 2')
    peppers_interp_by_8 = interpulation(interpulation(interpulation(peppers)))
    print('Saving interpolated peppers image (by 8)')
    cv2.imwrite('./Peppers_interpolated_by_8.jpg', peppers_interp_by_8)

    # 2.Equal Histogram:
    print('Loading leaf image')
    leaf = cv2.imread('./leafs.jpg', 0)
    # 2.a
    print('Calculating and plotting the histogram of the leaf image by the pixels values (0-255)')
    plot_hist(leaf, 'Leafs Histogram')
    # 2.b
    print('Stretching contrast of the leaf image')
    strech_leaf = stretch_contrast(leaf, 'Stretched Leafs')
    plot_hist(strech_leaf, 'Stretched Leafs Histogram')
    # 2.c
    print('Equalizing the histogram of the leaf image')
    equalize_leaf = equalize_hist(leaf, 'Equalized Leafs')
    plot_hist(equalize_leaf, 'Equalized Leafs Histogram')
    print('Done')
