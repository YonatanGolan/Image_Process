import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage,fftpack

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
    padded_img = np.zeros((n + 2 * padding_val, m + 2 * padding_val))
    padded_img[padding_val:padding_val + n, padding_val:padding_val + m] = img
    new_img = np.zeros((n, m))
    # Iterate over every pixel in the new image and apply the kernel
    # on the surrounding pixels in the original image
    for i in range(n):
        for j in range(m):
            new_img[i, j] = np.sum(padded_img[i:i + k, j:j + k] * kernel)
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

def sobel_filter(img):
    """ This function applies the horizontal sobel filter on the image
        as learned in the lecture
        input: img - 2D numpy array of the original image
        output: 2D numpy array of the filtered image"""

    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    grad_x = conv2d(img, kernel_x)
    return grad_x
def fft(img):
    """ This function applies the FFT on the image
        input: img - 2D numpy array of the original image
        output: 2D numpy array of the filtered image"""
    return fftpack.fftshift(fftpack.fft2(img))
def ifft(fft_img):
    """ This function applies the inverse FFT on the image
        input: fft_img - 2D numpy array of the FFT image
        output: 2D numpy array of the filtered image"""
    return fftpack.ifft2(fftpack.ifftshift(fft_img)).real
def display_mag_phase(fft_img, title = 'FFT'):
    """ This function displays the magnitude and the phase of the FFT
        input: fft_img - 2D numpy array of the FFT image"""
    plt.figure()
    plt.subplot(121), plt.imshow(np.log(1 + np.abs(fft_img))) # Log scaling for better visualization
    plt.title(f'Magnitude of {title}'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(np.angle(fft_img))
    plt.title(f'Phase of {title}'), plt.xticks([]), plt.yticks([])
    plt.savefig(f'./fft_{title}.jpg')  # Save the image - optional
    plt.show()

def display_mag(fft_img, title = 'FFT'):
    """ This function displays the magnitude of the FFT
        input: fft_img - 2D numpy array of the FFT image"""
    plt.imshow(np.log(1 + np.abs(fft_img)))
    plt.title(f'Magnitude of {title}')
    plt.savefig(f'./Magnitude_{title}.jpg')  # Save the image - optional
    plt.show()
def display_phase(fft_img, title = 'FFT'):
    """ This function displays the phase of the FFT
        input: fft_img - 2D numpy array of the FFT image"""
    plt.imshow(np.angle(fft_img))
    plt.title(f'Phase of {title}')
    plt.savefig(f'./phase_{title}.jpg')  # Save the image - optional
    plt.show()

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
    cv2.waitKey(0)
    print("Saving images after applying the directive filter")
    cv2.imwrite('./directive_img_I.jpg', directive_img_I)
    cv2.imwrite('./directive_img_I_n.jpg', directive_img_I_n)


    # Apply Gaussian filter on I_n.jpg
    gaussian_img = gaussian_filter(img_I_n, 2)
    # Save the image
    print("Saving image after applying the gaussian filter")
    cv2.imwrite('./I_dn.jpg', gaussian_img)
    # Display the image
    cv2.imshow('Gaussian Filter on I_n.jpg', gaussian_img)
    cv2.waitKey(0)

    # Apply Sobel filter on I_n.jpg
    sobel_img = sobel_filter(img_I_n)
    # Save the image
    print("Saving image after applying the sobel filter")
    cv2.imwrite('./I_dn2.jpg', sobel_img)
    # Display the image
    cv2.imshow('Sobel Filter on I_n.jpg', sobel_img)
    cv2.waitKey(0)

    #3a calculate the FFT of I.jpg and I_n.jpg and display the magnitude and the phase
    fft_img_I = fft(img_I)
    fft_img_I_n = fft(img_I_n)
    # Display the magnitude and the phase of the FFT
    print("Displaying the magnitude and the phase of the FFT")
    display_mag_phase(fft_img_I, 'I.jpg')
    display_mag_phase(fft_img_I_n, 'I_n.jpg')

    # 3b subtract the magnitude of the FFT of I_n.jpg from the magnitude of the FFT of I.jpg
    # and display the magnitude of the result
    print("Subtracting the magnitude of the FFT of I_n.jpg from the magnitude of the FFT of I.jpg")
    fft_img_diff = np.abs(np.abs(fft_img_I) - np.abs(fft_img_I_n))
    display_mag(fft_img_diff, 'I-I_n')

    #3c magnitude of chita.jpg and phase of zebra.jpg
    img_chita = cv2.imread('./chita.jpeg', 0)
    img_zebra = cv2.imread('./zebra.jpeg', 0)
    print("Displaying the magnitude of chita.jpeg and the phase of zebra.jpeg")
    fft_img_chita = fft(img_chita)
    fft_img_zebra = fft(img_zebra)
    display_mag(fft_img_chita, 'chita.jpeg')
    display_phase(fft_img_zebra, 'zebra.jpeg')

    #3d calculate the inverse FFT of magnitude of chita.jpg and phase of zebra.jpg
    # and display the result
    # Resize the images to have the same dimensions
    img_chita = cv2.resize(img_chita, (img_zebra.shape[1], img_zebra.shape[0]))
    print("Calculating the inverse FFT of magnitude of chita.jpg and phase of zebra.jpg as mixed image")
    fft_img_chita = fft(img_chita)
    fft_mixed_img = np.abs(fft_img_chita) * np.exp(1j * np.angle(fft_img_zebra))
    mixed_img = ifft(fft_mixed_img)
    plt.imshow(mixed_img, cmap='gray')
    plt.title('Mixed Image')
    plt.show()
    cv2.imwrite('./mixed_img.jpg', mixed_img)
    print("Done")

if __name__ == '__main__':
    main()