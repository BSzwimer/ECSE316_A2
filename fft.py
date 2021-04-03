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

    if mode == 2:
        filt = 0.12
        n_filt = 1 - filt
        ftt_img = fast_2D(im)
        h, w = ftt_img.shape
        lower_bound_h = int(filt*h)
        upper_bound_h = int(h*n_filt)
        lower_bound_w = int(w*filt)
        upper_bound_w = int(w*n_filt)
        ftt_img[lower_bound_h:upper_bound_h, :] = 0.0
        ftt_img[:,lower_bound_w:upper_bound_w] = 0.0

        denoised_img = IDFT_2D(ftt_img).real
        plt.subplot(1,2,1), plt.imshow(img, cmap = 'gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(1,2,2), plt.imshow(denoised_img, cmap = 'gray'), plt.xticks([]), plt.yticks([])
        plt.show()

    if mode==3:
        ftt_img = fast_2D(im)
        height, width = ftt_img.shape
        compession_levels = [19, 38, 57, 76, 95]
        img_list = []
        for i in range(5):
            opp = 100 - compession_levels[i]
            lower_bound = np.percentile(ftt_img, opp//2)
            upper_bound = np.percentile(ftt_img, 100 - opp//2)
            filtered = ftt_img * np.logical_or(ftt_img <= lower_bound, ftt_img >= upper_bound)
            inversed = IDFT_2D(filtered).real
            img_list.append(inversed)
        
        plt.subplot(2,3,1), plt.imshow(img, cmap = 'gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,2), plt.imshow(img_list[0], cmap = 'gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,3), plt.imshow(img_list[1], cmap = 'gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,4), plt.imshow(img_list[2], cmap = 'gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,5), plt.imshow(img_list[3], cmap = 'gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,6), plt.imshow(img_list[4], cmap = 'gray'), plt.xticks([]), plt.yticks([])

        plt.show()


def filter_mode_three(height, width, img, level):
    for i in range(width):
        for j in range(height):
            if (int(abs(img[j,i]))< level):
                img[j,i] = complex(0,0)

    return img



def IDFT_Naive(array):
    array = np.asarray(array,dtype=complex)
    N = array.shape[0]
    X = np.array([[np.exp(2j * np.pi * v * y / N) for v in range(N)] for y in range(N)])
    return 1/N * np.dot(X,array)



def IDFT_2D(input2DArray):
    input2DArray = np.asarray(input2DArray, dtype=complex)
    return np.array(list(map(lambda a: a / (len(input2DArray) * len(input2DArray[0])), IDFT_fast(IDFT_fast(input2DArray).T).T)))

def IDFT_fast(input2DArray):
    N = input2DArray.shape[1]
    if N <= 16:
        arr = []
        for i in range(input2DArray.shape[0]):
            arr.append(IDFT_Naive(input2DArray[i, :]))
        return np.array(arr)

    else:
        even = IDFT_fast(input2DArray[:, ::2])
        odd = IDFT_fast(input2DArray[:, 1::2])
        fact = []
        for _ in range(input2DArray.shape[0]):
            fact.append(np.exp(2j * np.pi * np.arange(N) / N))
        factor = np.array(fact)

        ans = []
        ans.append(even + np.multiply(factor[:, :int(N / 2)], odd))
        ans.append(even + np.multiply(factor[:, int(N / 2):], odd))
        return np.hstack(ans)

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
