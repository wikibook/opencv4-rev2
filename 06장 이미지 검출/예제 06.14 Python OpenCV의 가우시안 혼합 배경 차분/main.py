import cv2

capture = cv2.VideoCapture("basketball.mp4")

subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=16, detectShadows=True
)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), anchor=(-1, -1))

while True:
    ret, frame = capture.read()

    if not ret:
        break

    fgmask = subtractor.apply(frame)

    retval, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_OTSU)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=3)

    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    dst = cv2.hconcat((frame, fgmask))

    cv2.imshow("dst", dst)
    if cv2.waitKey(30) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()