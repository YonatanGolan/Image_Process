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

def sobel_filter(img):
    """ This function applies the sobel filter on the image
        input: img - 2D numpy array of the original image
        output: 2D numpy array of the filtered image"""
    # Compute the Sobel gradients in the x direction (Leroy's told me to do so)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

    # Compute the magnitude of the gradients
    grad = np.abs(grad_x)

    # Normalize to 8-bit scale
    grad = grad / grad.max() * 255

    return grad.astype(np.uint8)

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
    plt.subplot(121), plt.imshow(np.log(1 + np.abs(fft_img)), cmap='gray') # Log scaling for better visualization
    plt.title(f'Magnitude of {title}'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(np.angle(fft_img), cmap='gray')
    plt.title(f'Phase of {title}'), plt.xticks([]), plt.yticks([])
    plt.savefig(f'./fft_{title}.jpg')  # Save the image - optional
    plt.show()

def display_mag(fft_img, title = 'FFT'):
    """ This function displays the magnitude of the FFT
        input: fft_img - 2D numpy array of the FFT image"""
    plt.imshow(np.log(1 + np.abs(fft_img)), cmap='gray')
    plt.title(f'Magnitude of {title}')
    plt.savefig(f'./Magnitude_{title}.jpg')  # Save the image - optional
    plt.show()
def display_phase(fft_img, title = 'FFT'):
    """ This function displays the phase of the FFT
        input: fft_img - 2D numpy array of the FFT image"""
    plt.imshow(np.angle(fft_img), cmap='gray')
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
    # plt.imshow(directive_img_I, cmap='gray')
    # plt.title('Directive Filter on I.jpg')
    # plt.show()
    cv2.imshow('Directive Filter on I_n.jpg', directive_img_I_n)
    cv2.imwrite('./directive_img_I.jpg', directive_img_I)
    cv2.imwrite('./directive_img_I_n.jpg', directive_img_I_n)
    cv2.waitKey(0)

    # Apply Gaussian filter on I_n.jpg
    gaussian_img = gaussian_filter(img_I_n, 2)
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
    plt.imshow(sobel_img, cmap='gray')
    plt.title('Sobel Filter on I_n.jpg')
    plt.show()
    cv2.waitKey(0)

    #3a calculate the FFT of I.jpg and I_n.jpg and display the magnitude and the phase
    fft_img_I = fft(img_I)
    fft_img_I_n = fft(img_I_n)
    # Display the magnitude and the phase of the FFT
    display_mag_phase(fft_img_I, 'I.jpg')
    display_mag_phase(fft_img_I_n, 'I_n.jpg')

    # 3b subtract the magnitude of the FFT of I_n.jpg from the magnitude of the FFT of I.jpg
    # and display the magnitude of the result
    fft_img_diff = np.abs(np.abs(fft_img_I) - np.abs(fft_img_I_n))
    display_mag(fft_img_diff, 'I-I_n')

    #3c magnitude of chita.jpg and phase of zebra.jpg
    img_chita = cv2.imread('./chita.jpeg', 0)
    img_zebra = cv2.imread('./zebra.jpeg', 0)
    fft_img_chita = fft(img_chita)
    fft_img_zebra = fft(img_zebra)
    display_mag(fft_img_chita, 'chita.jpeg')
    display_phase(fft_img_zebra, 'zebra.jpeg')

    #3d calculate the inverse FFT of magnitude of chita.jpg and phase of zebra.jpg
    # and display the result
    # Resize the images to have the same dimensions
    img_chita = cv2.resize(img_chita, (img_zebra.shape[1], img_zebra.shape[0]))
    fft_img_chita = fft(img_chita)
    fft_mixed_img = np.abs(fft_img_chita) * np.exp(1j * np.angle(fft_img_zebra))
    mixed_img = ifft(fft_mixed_img)
    plt.imshow(mixed_img, cmap='gray')
    plt.title('Mixed Image')
    plt.show()
    cv2.imwrite('./mixed_img.jpg', mixed_img)


if __name__ == '__main__':
    main()