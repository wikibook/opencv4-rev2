import cv2
import numpy as np

capture = cv2.VideoCapture("car.mp4")
ret, prev_frame = capture.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(
    prev_gray, maxCorners=500, qualityLevel=0.1, minDistance=16, blockSize=7
)

while True:
    ret, next_frame = capture.read()
    if not ret or next_frame is None:
        break

    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, next_gray, prev_pts, None
    )

    good_prev = prev_pts[status == 1]
    good_next = next_pts[status == 1]

    for i, (next, prev) in enumerate(zip(good_next, good_prev)):
        x_next, y_next = next.astype(int).ravel()
        x_prev, y_prev = prev.astype(int).ravel()
        cv2.line(next_frame, (x_prev, y_prev), (x_next, y_next), (0, 255, 0), 2)
        cv2.circle(next_frame, (x_next, y_next), 5, (0, 255, 0), -1)

    prev_gray = next_gray.copy()
    prev_pts = good_next.reshape(-1, 1, 2)

    cv2.imshow("Optical Flow", next_frame)
    key = cv2.waitKey(22)
    if key == ord("q"):
        break

    elif key == ord("w"):
        add_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=500, qualityLevel=0.1, minDistance=16, blockSize=7
        )
        prev_pts = np.concatenate((prev_pts, add_pts), axis=0)

capture.release()
cv2.destroyAllWindows()