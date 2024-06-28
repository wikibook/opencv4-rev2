import cv2
import numpy as np

image = cv2.imread("image.jpg")
height, width = image.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
result = np.zeros((height, 256), dtype=np.uint8)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

for x, y in enumerate(hist):
    cv2.line(result, (x, height), (x, int(height - y[0])), 255)

dst = np.hstack([image[:, :, 0], result])
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()