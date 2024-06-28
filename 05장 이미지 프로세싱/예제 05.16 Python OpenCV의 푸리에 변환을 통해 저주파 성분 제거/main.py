import cv2
import numpy as np

def shift(img):
    axes = tuple(range(img.ndim))
    shift = [dim // 2 for dim in img.shape]
    return np.roll(img, shift, axes)

def fourier_spectrum(dft):
    spectrum = shift(dft)
    spectrum = np.log(cv2.magnitude(spectrum[:, :, 0], spectrum[:, :, 1]) + 1)
    spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return spectrum

src = cv2.imread("pears.jpg", cv2.IMREAD_GRAYSCALE)
dft = cv2.dft(np.float32(src), flags=cv2.DFT_COMPLEX_OUTPUT)
spectrum = fourier_spectrum(dft)

dft = shift(dft)  # np.fft.fftshift(dft)
d = 100
rows, cols = src.shape[:2]
cy, cx = rows // 2, cols // 2
mask = np.zeros((rows, cols, 2), np.uint8)
mask[cy - d : cy + d, cx - d : cx + d] = 1
dft *= mask

dft = shift(dft)  # np.fft.ifftshift(fshift)
idft = cv2.idft(dft)
dst = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
dst = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow("spectrum", spectrum)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
