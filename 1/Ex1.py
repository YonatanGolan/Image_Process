import numpy as np
import cv2
from matplotlib import pyplot as plt


def interpolation(img):  # 1.a
    """ This function does super resolution by bilinear interpolation
        img: 2D numpy array of the original image (n*m)
        return: 2D numpy array of the new image (2n*2m)"""
    h, w = img.shape
    new_h, new_w = h * 2, w * 2
    new_img = np.zeros((new_h, new_w))

    # Iterate over every pixel in the new image
    for i in range(new_h):
        for j in range(new_w):
            # Find the coordinates in the original image
            x = i / 2
            y = j / 2

            # Get the coordinates of the surrounding pixels
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, h - 1)
            y1 = min(y0 + 1, w - 1)

            # Calculate the differences
            dx0 = x - x0
            dx1 = x1 - x
            dy0 = y - y0
            dy1 = y1 - y

            # Get the pixel values
            top_left = img[x0, y0]
            top_right = img[x0, y1]
            bottom_left = img[x1, y0]
            bottom_right = img[x1, y1]

            if x0 == x1 and y0 == y1:
                new_img[i, j] = top_left  # no need to interpolate - take original pixel
                continue
            if x0 == x1:
                new_img[i, j] = top_left * dy1 + top_right * dy0  # interpolate only in the y direction
                continue
            if y0 == y1:
                new_img[i, j] = top_left * dx1 + bottom_left * dx0  # interpolate only in the x direction
                continue
            top = top_left * dy1 + top_right * dy0  # interpolate in the y direction
            bottom = bottom_left * dy1 + bottom_right * dy0
            new_img[i, j] = top * dx1 + bottom * dx0  # interpolate in the x direction
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
    peppers_interp_by_2 = interpolation(peppers)
    print('Saving interpolated peppers image (by 2)')
    cv2.imwrite('./Peppers_interpolated_by_2.jpg', peppers_interp_by_2)
    # 1.c
    print('Interpolating peppers image by 8 ')  # 2 times interpolation by 2(to already interpolated by 2 image)
    peppers_interp_by_8 = interpolation(interpolation(peppers_interp_by_2))
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
