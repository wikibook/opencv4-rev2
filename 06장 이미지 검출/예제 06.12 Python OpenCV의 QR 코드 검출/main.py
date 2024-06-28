import cv2

image = cv2.imread("qr-code.png")
detector = cv2.QRCodeDetector()
retval, decodedInfo, points, straightCode = detector.detectAndDecodeMulti(image)

for info, point in zip(decodedInfo, points):
    print(info)
    cv2.rectangle(image, tuple(point[0].astype(int)), tuple(point[2].astype(int)), (0, 255, 0), 4)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()