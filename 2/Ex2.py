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

    n, m = img.shape
    k = kernel.shape[0]
    padding_val = k // 2
    # Zero padding
    img = np.pad(img, padding_val)
    new_img = np.zeros((n, m))
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            new_img[i - 1, j - 1] = np.sum(img[i - 1:i + 2, j - 1:j + 2] * kernel)
    return new_img

def directive_filter(img):
    """ This function applies the directive filter on the image
        instead of using the kernel 1*3 and write more code,
         we will use the following kernel which is 3*3 and will give the same result:
        input: img - 2D numpy array of the original image
        output: 2D numpy array of the filtered image"""
    kernel = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    return conv2d(img, kernel)

def gaussian_filter(img, sigma=1):
    """ This function applies the gaussian filter on the image
        input: img - 2D numpy array of the original image
        output: 2D numpy array of the filtered image"""
    return ndimage.gaussian_filter(img, sigma)

def sobel_filter(img, axis=-1):
    """ This function applies the sobel filter on the image
        input: img - 2D numpy array of the original image
        output: 2D numpy array of the filtered image"""
    # Compute the Sobel gradients in the x and y direction
    grad = ndimage.sobel(img)

    # Normalize to 8-bit scale
    grad = grad / grad.max() * 255

    return grad.astype(np.uint8)

def main():
    print("Loading image I.jpg")
    img_I = cv2.imread('./I.jpg', 0)

    print("Loading image I_n.jpg")
    img_I_n = cv2.imread('./I_n.jpg', 0)

    # Apply clipping filter on both images
    directive_img_I = directive_filter(img_I)
    directive_img_I_n = directive_filter(img_I_n)

    # Display the images
    cv2.imshow('Directive Filter on I.jpg', directive_img_I)
    cv2.imshow('Directive Filter on I_n.jpg', directive_img_I_n)
    cv2.imwrite('./directive_img_I.jpg', directive_img_I)
    cv2.imwrite('./directive_img_I_n.jpg', directive_img_I_n)
    cv2.waitKey(0)

    # Apply Gaussian filter on I_n.jpg
    gaussian_img = gaussian_filter(img_I_n, 1)
    # Save the image
    print ("Saving the gaussian image")
    cv2.imwrite('./I_dn.jpg', gaussian_img)
    # Display the image
    cv2.imshow('Gaussian Filter on I_n.jpg', gaussian_img)
    cv2.waitKey(0)

    # Apply Sobel filter on I_n.jpg
    sobel_img = sobel_filter(img_I_n)
    # Save the image
    print ("Saving the sobel image")
    cv2.imwrite('./I_dn2.jpg', sobel_img)
    # Display the image
    cv2.imshow('Sobel Filter on I_n.jpg', sobel_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()