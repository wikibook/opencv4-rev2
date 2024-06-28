import cv2

capture = cv2.VideoCapture("star.mp4")
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoWriter = cv2.VideoWriter()
isWrite = False

while True:
    ret, frame = capture.read()

    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.open("star.mp4")

    cv2.imshow("VideoFrame", frame)
    key = cv2.waitKey(33)

    if key == 4:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        videoWriter.open("video.avi", fourcc, 30, (width, height), True)
        isWrite = True

    elif key == 24:
        videoWriter.release()
        isWrite = False

    elif key == ord("q"):
        break

    if isWrite == True:
        videoWriter.write(frame)

videoWriter.release()
capture.release()
cv2.destroyAllWindows()