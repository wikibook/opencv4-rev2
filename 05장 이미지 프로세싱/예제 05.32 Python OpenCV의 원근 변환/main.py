import cv2
import numpy as np

src = cv2.imread("clouds.jpg")
height, width, _ = src.shape

src_pts = np.array(
    [
        [0, 0],
        [0, height],
        [width, height],
        [width, 0]
    ],
    dtype=np.float32
)
dst_pts = np.array(
    [
        [300, 300],
        [0, height - 200],
        [width - 100, height - 100],
        [900, 200]
    ],
    dtype=np.float32,
)

M = cv2.getPerspectiveTransform(src_pts, dst_pts)
dst = cv2.warpPerspective(src, M, (width, height), borderValue=(255, 255, 255, 0))

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()