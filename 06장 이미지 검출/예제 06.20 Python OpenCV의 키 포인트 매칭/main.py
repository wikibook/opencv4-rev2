import cv2

query = cv2.imread("query.jpg")
train = cv2.imread("train.jpg")
query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
train_gray = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB.create(nfeatures=5000)
kp1, des1 = orb.detectAndCompute(query_gray, None)
kp2, des2 = orb.detectAndCompute(train_gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

count = 100
for i in matches[:count]:
    idx = i.queryIdx
    x1, y1 = kp1[idx].pt
    cv2.circle(query, (int(x1), int(y1)), 3, (0, 0, 255), 3)

flag = (cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS | cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
result = cv2.drawMatches(query, kp1, train, kp2, matches[:count], None, flags=flag)

cv2.imshow("Matching", result)
cv2.waitKey(0)
cv2.destroyAllWindows()