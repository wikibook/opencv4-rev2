import cv2

src = cv2.imread("tomato.jpg")
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)

orange = cv2.inRange(hsv, (8, 100, 100), (20, 255, 255))
blue = cv2.inRange(hsv, (110, 100, 100), (130, 255, 255))
mix_color = cv2.addWeighted(orange, 1.0, blue, 1.0, 0.0)

mix = cv2.bitwise_and(hsv, hsv, mask=mix_color)
mix = cv2.cvtColor(mix, cv2.COLOR_HSV2BGR)

cv2.imshow("mix", mix)
cv2.waitKey(0)
cv2.destroyAllWindows()