import cv2
import numpy as np
import matplotlib.pyplot as plt


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    lenIn = inSignal.size
    lenKer = kernel1.size
    ans = np.zeros(lenIn + lenKer - 1)
    for i in np.arange(lenIn):
        for j in np.arange(lenKer):
            ans[i + j] += inSignal[i]*kernel1[j]
    print("ans = ", ans)
    lenAns = ans.size
    diff = lenAns - lenIn
    result = np.zeros(lenAns - diff)
    print("diff/2 = ", int(diff/2))
    if int(diff % 2) == 0:
        j = 0
        for i in range(int(diff/2), lenAns - int(diff/2)):
            result[j] = ans[i]
            j += 1
        print("result = ", result)
    elif int(diff % 2) != 0:
        j = 0
        for i in range(int(diff/2), lenAns - int(diff/2) - 1):
            result[j] = ans[i]
            j += 1
        print("resultOdd = ", result)
    return result


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    kernel = np.copy(kernel2)
    rows = inImage.shape[0]
    cols = inImage.shape[1]
    kRows = kernel.shape[0]
    kCols = kernel.shape[1]
    kCenterX = kernel.shape[0]//2
    kCenterY = kernel.shape[1]//2
    ans = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            sum = 0
            for m in range(kRows):
                mm = kRows - 1 - m
                for n in range(kCols):
                    nn = kCols - 1 - n
                    # index of input signal, used for checking boundary
                    ii = i - kCenterX + m
                    jj = j - kCenterY + n
                    # ignore input samples which are out of bound
                    if (ii>=0) and (ii<rows) and (jj>=0) and (jj<cols):
                        sum += inImage[ii][jj]*kernel[m][n]
            ans[i][j] = int(float(abs(sum)) + float(0.5))
    print(ans)
    return ans


# def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
#     kernel = np.copy(kernel2)
#     row = inImage.shape[0]
#     col = inImage.shape[1]
#     m = kernel.shape[0]
#     n = kernel.shape[1]
#     new = np.zeros((row+m-1, col+n-1))
#     m = kernel.shape[0]//2
#     n = kernel.shape[1]//2
#     filtered_img = np.zeros(inImage.shape)
#     new[m:new.shape[0]-m, n:new.shape[1]-n] = inImage
#     for i in range(m, new.shape[0]-m):
#         for j in range(n, new.shape[1]-n):
#             temp = new[i-m:i+m+1, j-m:j+m+1]
#             result = temp*kernel
#             filtered_img[i-m, j-n] = result.sum()
#     return filtered_img


def convDerivative(inImage: np.ndarray) -> np.ndarray:
    ker1 = np.array([[-1, 0, 1]])
    ker2 = np.array([[-1], [0], [1]])
    # Ix = conv2D(inImage, ker1)
    # Iy = conv2D(inImage, ker2)
    Ix = cv2.filter2D(inImage, cv2.CV_8U, ker1)
    Iy = cv2.filter2D(inImage, cv2.CV_8U, ker2)
    cv2.imshow('imx', Ix)
    cv2.waitKey(0)
    cv2.imshow('imx', Iy)
    cv2.waitKey(0)
    Ix_n = cv2.normalize(Ix.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    Iy_n = cv2.normalize(Iy.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    mag = cv2.magnitude(Ix_n, Iy_n)
    cv2.imshow('ma', mag)
    cv2.waitKey(0)
    return mag


# Q3
def blurImage1(inImage: np.ndarray, kernelSize: np.ndarray) -> np.ndarray:
    kernel = gaussianKernel(kernelSize)
    grey_im = cv2.cvtColor(inImage, cv2.COLOR_BGR2GRAY)
    grey_im = np.pad(grey_im, pad_width=1, mode='constant', constant_values=0)
    blurImage = conv2D(grey_im, kernel)
    # blurImage = cv2.filter2D(grey_im, -1, kernel)
    cv2.imshow('figur2', blurImage)
    cv2.waitKey(0)
    return blurImage


def blurImage2(inImage: np.ndarray, kernelSize: np.ndarray) -> np.ndarray:
    # example for output
    gaussianBluring(inImage, kernelSize)
    # implementation using cv2
    t1 = cv2.getGaussianKernel(kernelSize[0], 1.0)
    t2 = cv2.getGaussianKernel(kernelSize[1], 1.0)
    kernel = np.dot(t1, t2.T)
    # print("blurimage2 kernel = ", kernel)
    blurImage = cv2.filter2D(inImage, -1, kernel)
    cv2.imshow('figure1', blurImage)
    cv2.waitKey(0)
    return blurImage


def gaussianBluring(inImage: np.ndarray, kernelSize: np.ndarray):
    t = (kernelSize[0], kernelSize[1])
    output = cv2.GaussianBlur(inImage, t, 1.0, None, 1.0)
    cv2.imshow('mm', output)
    cv2.waitKey(0)


def gaussianKernel(kernelSize: np.ndarray) -> np.ndarray:
    m = kernelSize[0]
    n = kernelSize[1]
    shape = (m, n)
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / 2)
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def edgeDetectionSobel(I: np.ndarray) -> (np.ndarray, np.ndarray):
    # ####### OpenCV implementation ####### #
    h = cv2.Sobel(I, -1, 0, 1, -1)
    v = cv2.Sobel(I, -1, 1, 0, -1)
    # Ix = np.dot(h, h)           # this is one from tow method to calculate magnitude
    # Iy = np.dot(v, v)
    Ix_n = cv2.normalize(h.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    Iy_n = cv2.normalize(v.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    Ix_n = np.abs(Ix_n)
    Iy_n = np.abs(Iy_n)
    t_img = cv2.magnitude(Ix_n, Iy_n)
    cv2.imshow('edgeImage2', t_img)
    cv2.waitKey(0)
    # ####### My implementation ####### #
    # By the book of "open university"
    im_copy = np.copy(I)
    Gx = np.zeros(im_copy.shape)
    Gy = np.zeros(im_copy.shape)
    for m in range(1, Gx.shape[0] - 1):
        for n in range(1, Gx.shape[1] - 1):
            Gx[m, n] = 1/8*(im_copy[m-1, n-1] + 2*im_copy[m-1, n] + im_copy[m-1, n+1] -
                            im_copy[m+1, n-1] - 2*im_copy[m+1, n] - im_copy[m+1, n+1])
            Gy[m, n] = 1/8*(im_copy[m-1, n-1] + 2*im_copy[m, n-1] + im_copy[m+1, n-1] -
                            im_copy[m-1, n+1] - 2*im_copy[m, n+1] - im_copy[m+1, n+1])
    Gx = np.abs(Gx)
    Gy = np.abs(Gy)
    Gx_n = cv2.normalize(Gx.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    Gy_n = cv2.normalize(Gy.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    ans = cv2.magnitude(Gx_n, Gy_n)
    cv2.imshow('edgeImage1', ans)
    cv2.waitKey(0)
    return ans, t_img


def edgeDetectionZeroCrossingSimple(I: np.ndarray) -> (np.ndarray, np.ndarray):
    # ####### OpenCV implementation ####### #
    laplacian = cv2.Laplacian(I, cv2.CV_64F)
    cv2.imshow('edgeImage2', laplacian)
    cv2.waitKey(0)
    # ####### My implementation ####### #
    # By the book of "open university"
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    ans = cv2.filter2D(I, -1, kernel)
    cv2.imshow('edgeImage1', ans)
    cv2.waitKey(0)
    return ans, laplacian



# a = np.array([1, 2, 3])
# b = np.array([0, 1, 0.5])
# conv1 = conv1D(a, b)
# r = np.convolve(a, b, mode='same')
# print(r)

# aa = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.int32)
# bb = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.int32)
# print(aa)
# print(bb)
# conv2D(aa, bb)

im = cv2.imread('m.png')
# cv2.imshow('lena', im)
# cv2.waitKey(0)
grey_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# cv2.imshow('lena', grey_im)
# cv2.waitKey(0)
# ans_mag = convDerivative(grey_im)

# blurImage2(im, np.array([5, 5]))
# blurImage1(im, np.array([5, 5]))

retval, imgmask = cv2.threshold(grey_im, 120, 255, cv2.THRESH_BINARY)
# cv2.imshow('figure1', imgmask)
# cv2.waitKey(0)
# edgeDetectionSobel(imgmask)
edgeDetectionZeroCrossingSimple(imgmask)
