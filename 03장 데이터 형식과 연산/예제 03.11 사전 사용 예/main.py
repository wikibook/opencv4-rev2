import cv2

# 실행을 위한 추가 코드
img = cv2.imread("tomato.jpg")

COLOR_CODE = {
    "yellow": [(20, 100, 100), (33, 255, 255)],
    "green": [(41, 100, 100), (70, 255, 255)],
    "blue": [(90, 100, 100), (150, 100, 255)],
}

result = cv2.inRange(img, COLOR_CODE["yellow"][0], COLOR_CODE["yellow"][1])

# 실행을 위한 추가 코드
cv2.imshow("result", result)
cv2.waitKey(0)