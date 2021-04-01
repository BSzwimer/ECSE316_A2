import argparse
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def mode(mode, image_path):
    #read in image

    img = plt.imread(image_path).astype(float)

    height,width= img.shape
    #pad the image
    new_width = image_pad(width)
    new_height = image_pad(height)

    new_size = (new_width, new_height)
    im = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


    if mode == 1:
        ftt_img = fast_2D(im)
        plt.figure("Mode 1")
        plt.subplot(1,2,1), plt.imshow(img, cmap = 'gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(1,2,2), plt.imshow(np.abs(ftt_img), norm=colors.LogNorm()), plt.xticks([]), plt.yticks([])
        plt.show()

def fast_2D(img):
    img = np.asarray(img, dtype=complex)
    height, width = img.shape
    ans = np.zeros((height, width), dtype=complex)

    for column in range(width):
        ans[:, column] = FFT_1D(img[:,column])

    for row in range(height):
        ans[row, :] = FFT_1D(ans[row, :])

    return ans

def FFT_1D(array):
    array = np.asarray(array, dtype=complex)
    N = array.shape[0]

    if N == 16:
        return DFT_Naive(array)
    else:
        even = FFT_1D(array[::2])
        odd = FFT_1D(array[1::2])
        n_idx = np.arange(N)
        f = np.exp(-2j * np.pi * n_idx / N)

        return np.concatenate([even + f[:N // 2] * odd,
          
                               even + f[N // 2:] * odd])



def DFT_Naive(array):

    array = np.asarray(array, dtype=complex)
    N = array.shape[0]

    X = np.array([[np.exp(-2j * np.pi * v * y / N) for v in range(N)] for y in range(N)])
    
    return np.dot(X,array)

def IDFT_Naive(array):
    array = np.asarray(array,dtype=complex)
    N = array.shape[0]

    X = np.array([[np.exp(2j * np.pi * v * y / N) for v in range(N)] for y in range(N)])
    return 1/N * np.dot(X,array)



def image_pad(dim):
    if dim == 0:
        return 1
    else:
        return 2 ** (dim-1).bit_length()

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=1, dest='mode', action='store')
    parser.add_argument('-i', type=str, default='moonlanding.png', dest='image_path', action='store')

    mode(parser.parse_args().mode, parser.parse_args().image_path)




if __name__ == "__main__":
    __main__()
