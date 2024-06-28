import cv2
import numpy as np

src = cv2.imread("egg.jpg")
data = src.reshape(-1, 3).astype(np.float32)

K = 3
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.001)
retval, best_labels, centers = cv2.kmeans(
    data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

centers = centers.astype(np.uint8)
dst = centers[best_labels].reshape(src.shape)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()