import cv2

src = cv2.imread("ferris-wheel.jpg")
gaussian_pyramid = []
laplacian_pyramid = []
sizes = []

num_levels = 4
temp = src.copy()
for i in range(num_levels):
    down = cv2.pyrDown(temp)
    gaussian_pyramid.append(down)
    temp = down.copy()
    sizes.append((down.shape[1], down.shape[0]))

for i in range(num_levels - 1):
    up = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=sizes[i])
    laplacian = cv2.subtract(gaussian_pyramid[i], up)
    laplacian_pyramid.append(laplacian)

cv2.imshow("gaussianPyramid_0", gaussian_pyramid[0])
cv2.imshow("laplacianPyramid_0", laplacian_pyramid[0])
cv2.waitKey(0)
cv2.destroyAllWindows()