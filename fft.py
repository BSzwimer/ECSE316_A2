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
        ans[:, column] = fast_1D(img[:,column])

    for row in range(height):
        ans[row, :] = fast_1D(ans[row, :])

    return ans

def fast_1D(img):
    img = np.asarray(img, dtype=complex)
    height = img.shape[0]


    if height <= 16:
        return slow_1D(img)
    else:
        ans = np.zeros(height, dtype=complex)
        odd = fast_1D(img[1::2])
        even = fast_1D(img[::2])

        half = height//2
        for h in range(height):
            ans[h] = even[h % half] + np.exp(-2j * np.pi * h / height) * odd[h % half]

        return ans

def slow_1D(img):

    img = np.asarray(img, dtype=complex)
    height = img.shape[0]
    ans = np.zeros(height, dtype=complex)

    for i in range(height):
        for j in range(height):
            ans[i] += img[j] * np.exp(-2j * np.pi * i * j / height)

    return ans

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
