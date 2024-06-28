import cv2
import numpy as np

src = cv2.imread("gerbera.jpg", cv2.IMREAD_REDUCED_GRAYSCALE_2)
kernel = np.array([
    [-1, -2, -1],
    [-2, 12, -2],
    [-1, -2, -1]
])
dst = cv2.filter2D(src, -1, kernel)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()