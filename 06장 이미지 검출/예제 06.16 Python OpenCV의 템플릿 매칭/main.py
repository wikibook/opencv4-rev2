import cv2

src = cv2.imread("hats.jpg")
templ = cv2.imread("hat.jpg")
dst = src.copy()

result = cv2.matchTemplate(src, templ, cv2.TM_SQDIFF_NORMED)

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

x, y = minLoc
h, w = templ.shape[:2]
dst = cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 4)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()