import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage

def conv2d(img, kernel):
    """ This function performs 2D convolution between an image and a kernel
        input: img - 2D numpy array of the original image,
               kernel - 2D numpy array of the kernel, size k*k
        output: 2D numpy array of the convolved image which in same
                size as the original image using zero padding."""
    # Zero padding
    img = np.pad(img, 1)
    n, m = img.shape
    k = kernel.shape[0]
    new_img = np.zeros((n - 2, m - 2))
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            new_img[i - 1, j - 1] = np.sum(img[i - 1:i + 2, j - 1:j + 2] * kernel)
    return new_img

def directive_filter(img):
    """ This function applies the directive filter on the image
        input: img - 2D numpy array of the original image
        output: 2D numpy array of the filtered image"""
    kernel = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    return conv2d(img, kernel)

def gaussian_filter(img, sigma=1):
    """ This function applies the gaussian filter on the image
        input: img - 2D numpy array of the original image
        output: 2D numpy array of the filtered image"""
    return ndimage.gaussian_filter(img, sigma)

def sobel_filter(img):
    """ This function applies the sobel filter on the image
        input: img - 2D numpy array of the original image
        output: 2D numpy array of the filtered image"""
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return conv2d(img, kernel)

def main():
    # load image I.jpg
    img = cv2.imread('./I.jpg', 0)
    directive_img = directive_filter(img)
    # Save the image
    cv2.imwrite('./Directive_I.jpg', directive_img)
    gaussian_img = gaussian_filter(img)
    # Save the image
    cv2.imwrite('./Gaussian_I.jpg', gaussian_img)
    sobel_img = sobel_filter(img)
    # Save the image
    cv2.imwrite('./Sobel_I.jpg', sobel_img)

    # plt.figure()
    # plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    # plt.subplot(222), plt.imshow(directive_img, cmap='gray'), plt.title('Directive Filter')
    # plt.subplot(223), plt.imshow(gaussian_img, cmap='gray'), plt.title('Gaussian Filter')
    # plt.subplot(224), plt.imshow(sobel_img, cmap='gray'), plt.title('Sobel Filter')
    # plt.show()

if __name__ == '__main__':
    main()